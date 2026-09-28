from datetime import datetime

from rag_eval.artifacts.artifact_lineage import (
    ArtifactNode,
    ArtifactRef,
    ArtifactType,
)
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.artifact_store import ArtifactStore
from rag_eval.config import PipelineConfig
from rag_eval.evaluator.evaluate_pipeline import evaluate_pipeline
from rag_eval.indexing.read_md_docs import read_md_docs
from rag_eval.indexing.run_chunks_embedder import run_chunks_embedder
from rag_eval.indexing.run_splitter import run_splitter
from rag_eval.pipeline.persist_pipeline_run_config import persist_pipeline_run_config
from rag_eval.questions.run_candidate_chunks_retrieval import (
    run_candidate_chunks_retrieval,
)
from rag_eval.questions.run_candidates_judge import run_candidates_judge
from rag_eval.questions.run_questions_generator import run_questions_generator
from rag_eval.questions.run_questions_quality_filter import run_questions_quality_filter
from rag_eval.questions.run_questions_quality_judge import run_questions_quality_judge
from rag_eval.simulation.run_retrievals_generation import (
    run_retrievals_generation,
)
from rag_eval.simulation.summarize_results import summarize_results
from rag_eval.tests.run_tests_preparation import run_tests_preparation

from bot.config.embedder import LocalEmbedderConfig, SupportedLocalEmbedder
from bot.doc_qa.retrieval.embedders.factory import create_embedder


class EvaluationPipeline:
    def __init__(
        self,
        *,
        config: PipelineConfig,
        artifact_store: ArtifactStore,
        runner: ArtifactStepExecutor,
    ):
        self._config = config
        self._artifact_store = artifact_store
        self._runner = runner

    def run(self):
        root = ArtifactNode(artifact_id="root", artifact_type=ArtifactType.ROOT, children=[])

        print("Reading markdown documents...")
        reader_art = read_md_docs(
            root=ArtifactRef(artifact_node=root, data=None), config=self._config.md_docs
        )

        print("Splitting markdown documents into chunks...")
        splitter_art = run_splitter(
            md_docs=reader_art, config=self._config.md_chunking, runner=self._runner
        )

        print("Embedding document chunks...")
        embeddings_art = run_chunks_embedder(
            chunks=splitter_art, config=self._config.docs_embeddings, runner=self._runner
        )

        print("Generating question collection...")
        question_collection_art = run_questions_generator(
            doc_chunks=splitter_art, config=self._config.question_generator, runner=self._runner
        )

        print("Picking high-quality questions...")
        judged_questions = run_questions_quality_judge(
            question_collection=question_collection_art,
            config=self._config.question_quality_judge,
            runner=self._runner,
        )
        hq_questions_collection = run_questions_quality_filter(
            questions=question_collection_art,
            question_judgements=judged_questions,
            runner=self._runner,
            config=self._config.filter_quality_config,
        )

        print("Embedding document chunks with Jina embeddings...")
        jina_embeddings_art = run_chunks_embedder(
            chunks=splitter_art,
            config=LocalEmbedderConfig(model=SupportedLocalEmbedder.JINA_EMBEDDINGS_V3),
            runner=self._runner,
        )

        print("Embedding document chunks with Qwen embeddings...")
        qwen_embeddings_art = run_chunks_embedder(
            chunks=splitter_art,
            config=LocalEmbedderConfig(model=SupportedLocalEmbedder.QWEN3_EMBEDDING_4B),
            runner=self._runner,
        )

        print("Retrieval of candidate chunks...")
        candidate_chunks_art = run_candidate_chunks_retrieval(
            question_collection=hq_questions_collection,
            doc_chunks=splitter_art,
            jina_embeddings=jina_embeddings_art,
            qwen_embeddings=qwen_embeddings_art,
            config=self._config.candidate_answers_generator,
            runner=self._runner,
        )

        print("Judging candidate chunks...")
        judgement = run_candidates_judge(
            question_collection=hq_questions_collection,
            candidate_retrieval=candidate_chunks_art,
            config=self._config.judge,
            runner=self._runner,
        )

        print("Preparing tests...")
        tests = run_tests_preparation(
            question_collection=hq_questions_collection,
            candidate_retrieval_judgements=judgement,
            runner=self._runner,
            config=self._config.test_preparation,
        )

        precomputed_retrievals = run_retrievals_generation(
            test_suite=tests,
            embedder=create_embedder(self._config.docs_embeddings),
            embedded_doc_chunks=embeddings_art,
            max_top_k=max(config.retriever_top_k for config in self._config.evaluation.k_configs),
            runner=self._runner,
            config=self._config.runner,
        )

        # config=self._config.eval_retrieval,
        print("Evaluating the pipeline...")
        evaluation = evaluate_pipeline(
            test_suite=tests.data,
            precomputed_retrievals=precomputed_retrievals.data,
            configs=self._config.evaluation,
        )

        date_now_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        print("Summarizing results...")
        summary = summarize_results(
            evaluation=evaluation,
            artifact_id=f"summary_{date_now_str}",
            artifact_store=self._artifact_store,
        )
        print(summary)
        persist_pipeline_run_config(
            artifact_id=f"run_{date_now_str}", root=root, artifact_store=self._artifact_store
        )
