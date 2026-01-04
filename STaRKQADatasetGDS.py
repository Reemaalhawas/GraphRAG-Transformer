import os
import yaml
import numpy as np
import pandas as pd
import torch

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from graphdatascience import GraphDataScience
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from compute_metrics import compute_intermediate_metrics


# =========================================================
# ------------------- Neo4j Helpers -----------------------
# =========================================================

def get_nodes_by_vector_search(
    query_embedding: np.ndarray,
    k_nodes: int,
    driver: Driver
) -> list[str]:
    """Retrieve top-k nodes using Neo4j vector index."""
    with driver.session() as session:
        res = session.run(
            """
            CALL db.index.vector.queryNodes($index, $k, $query_embedding)
            YIELD node
            RETURN elementId(node) AS nodeId
            """,
            index="text_embeddings",
            k=k_nodes,
            query_embedding=query_embedding.tolist(),
        )
        return [r["nodeId"] for r in res]


def cypher_retrieval(node_ids: list[str], driver: Driver) -> pd.DataFrame:
    """Retrieve edges between nodes."""
    with driver.session() as session:
        res = session.run(
            """
            UNWIND $nodeIds AS nodeId
            MATCH (m)-[r]-(n)
            WHERE elementId(m) = nodeId
            RETURN
                elementId(m) AS sourceNodeId,
                elementId(n) AS targetNodeId,
                type(r) AS relationshipType
            """,
            nodeIds=node_ids,
        )

        df = pd.DataFrame([dict(r) for r in res])

    # Ensure expected columns exist
    return df if not df.empty else pd.DataFrame(
        columns=["sourceNodeId", "targetNodeId", "relationshipType"]
    )


def get_textual_nodes(node_ids: list[str], driver: Driver) -> pd.DataFrame:
    """Fetch textual node attributes."""
    with driver.session() as session:
        res = session.run(
            """
            UNWIND $nodeIds AS nodeId
            MATCH (n)
            WHERE elementId(n) = nodeId
            RETURN
                elementId(n) AS nodeId,
                n.name AS name,
                coalesce(n.details, n.description, "") AS description,
                n.embedding AS embedding
            """,
            nodeIds=node_ids,
        )

        df = pd.DataFrame([dict(r) for r in res])

    return df if not df.empty else pd.DataFrame(
        columns=["nodeId", "name", "description", "embedding"]
    )


def get_textual_edges(edge_pairs: list[tuple[str, str]], driver: Driver) -> pd.DataFrame:
    """Fetch textual edge attributes."""
    if not edge_pairs:
        return pd.DataFrame(columns=["src", "edge_attr", "dst"])

    with driver.session() as session:
        res = session.run(
            """
            UNWIND $pairs AS pair
            MATCH (s)-[e]->(t)
            WHERE elementId(s) = pair[0] AND elementId(t) = pair[1]
            RETURN
                elementId(s) AS src,
                type(e) AS edge_attr,
                elementId(t) AS dst
            """,
            pairs=edge_pairs,
        )

        df = pd.DataFrame([dict(r) for r in res])

    return df if not df.empty else pd.DataFrame(columns=["src", "edge_attr", "dst"])


# =========================================================
# ------------------- Utilities ---------------------------
# =========================================================

def textualize_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> str:
    nodes_df = nodes_df.copy()
    nodes_df["description"] = nodes_df["description"].fillna("")
    nodes_df["node_attr"] = nodes_df.apply(
        lambda r: f"name: {r['name']}, description: {r['description']}", axis=1
    )

    nodes_csv = nodes_df[["nodeId", "node_attr"]].to_csv(index=False)
    edges_csv = edges_df.to_csv(index=False)
    return nodes_csv + "\n" + edges_csv


def assign_node_prizes(nodes_df: pd.DataFrame, top_nodes: list[str]):
    prize_map = {n: len(top_nodes) - i for i, n in enumerate(top_nodes)}
    nodes_df["nodePrize"] = nodes_df["nodeId"].map(prize_map).fillna(0)


def assign_edge_costs(edges_df: pd.DataFrame):
    edges_df["edgeCost"] = 0.5


# =========================================================
# ------------------- Dataset -----------------------------
# =========================================================

class STaRKQADataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        raw_dataset: Dataset,
        retrieval_config_version: int,
        algo_config_version: int,
        split: str = "train",
        force_reload: bool = False,
    ):
        self.split = split
        self.raw_dataset = raw_dataset
        self.retrieval_config_version = retrieval_config_version
        self.algo_config_version = algo_config_version

        self.query_embedding_dict = torch.load(
            os.path.join(
                os.path.dirname(__file__),
                "data-loading/emb/prime/text-embedding-ada-002/query/query_emb_dict.pt",
            )
        )

        super().__init__(root, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"{self.split}_data.pt"]

    def process(self):
        load_dotenv("db.env", override=True)

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        pwd = os.getenv("NEO4J_PASSWORD")

        dataframe = self.raw_dataset.data.loc[self.raw_dataset.indices]
        answer_ids = {i: eval(r[2]) for i, r in dataframe.iterrows()}

        with open(f"configs/retrieval_config_v{self.retrieval_config_version}.yaml") as f:
            retrieval_cfg = yaml.safe_load(f)

        base_path = f"base_subgraphs/v{self.retrieval_config_version}"
        os.makedirs(base_path, exist_ok=True)
        base_file = f"{base_path}/{self.split}_base.pt"

        if os.path.exists(base_file):
            base_subgraphs = torch.load(base_file)
        else:
            base_subgraphs = {}
            print("🔎 Building base subgraphs...")

            for idx, (qid, _, _) in tqdm(dataframe.iterrows()):
                emb = self.query_embedding_dict[qid].numpy()[0]
                with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
                    nodes = get_nodes_by_vector_search(
                        emb, retrieval_cfg["k_nodes"] * 25, driver
                    )[: retrieval_cfg["k_nodes"]]

                    edges_df = cypher_retrieval(nodes, driver)

                # Robust column detection
                src_col = next(c for c in edges_df.columns if "source" in c.lower())
                dst_col = next(c for c in edges_df.columns if "target" in c.lower())

                node_ids = np.unique(
                    np.concatenate([edges_df[src_col], edges_df[dst_col]])
                )

                base_subgraphs[idx] = (
                    pd.DataFrame({"nodeId": node_ids}),
                    edges_df,
                )

            torch.save(base_subgraphs, base_file)
            compute_intermediate_metrics(answer_ids, {
                k: v[0]["nodeId"].tolist() for k, v in base_subgraphs.items()
            })

        # ---------------- PCST / Fallback ----------------

        with open(f"configs/algo_config_v{self.algo_config_version}.yaml") as f:
            algo_cfg = yaml.safe_load(f)

        data_list = []

        print("🧠 Computing final graphs...")
        for idx, (qid, prompt, _) in tqdm(dataframe.iterrows()):
            emb = self.query_embedding_dict[qid].numpy()[0]
            nodes_df, edges_df = base_subgraphs[idx]

            with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
                top_nodes = get_nodes_by_vector_search(
                    emb, algo_cfg["topk_nodes"], driver
                )

            assign_node_prizes(nodes_df, top_nodes)
            assign_edge_costs(edges_df)

            with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
                nodes_txt = get_textual_nodes(top_nodes, driver)
                edges_txt = get_textual_edges(
                    list(zip(edges_df["sourceNodeId"], edges_df["targetNodeId"])),
                    driver,
                )
                answers = get_textual_nodes(answer_ids[idx], driver)["name"].tolist()

            desc = textualize_graph(nodes_txt, edges_txt)
            x = torch.tensor(nodes_txt["embedding"].tolist(), dtype=torch.float)

            node_map = {nid: i for i, nid in enumerate(nodes_txt["nodeId"])}
            edge_index = torch.tensor(
                [
                    (node_map[s], node_map[t])
                    for s, t in zip(edges_df["sourceNodeId"], edges_df["targetNodeId"])
                    if s in node_map and t in node_map
                ],
                dtype=torch.long,
            ).T

            data_list.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                    question=f"Question: {prompt}\nAnswer:",
                    label="|".join(answers).lower(),
                    desc=desc,
                )
            )

        self.save(data_list, self.processed_paths[0])

