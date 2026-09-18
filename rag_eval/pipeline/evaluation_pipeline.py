
from datetime import datetime

from bot.config.embedder import LocalEmbedderConfig, SupportedLocalEmbedder
from bot.factories.embedder import create_embedder
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.artifact_store import ArtifactStore
from rag_eval.artifacts.ArtifactLineage import (
    ArtifactNode,
    ArtifactRef,
    ArtifactType,
)
from rag_eval.artifacts.steps.embed_doc_chunks import embed_doc_chunks
from rag_eval.artifacts.steps.filter_judged_question_collection import (
    filter_judged_question_collection,
)
from rag_eval.artifacts.steps.generate_candidate_chunks import generate_candidate_chunks
from rag_eval.artifacts.steps.generate_questions_collection import (
    generate_question_collection,
)
from rag_eval.artifacts.steps.judge_chunks import judge_candidates
from rag_eval.artifacts.steps.judge_question_collection import judge_question_collection
from rag_eval.artifacts.steps.persist_run_config import persist_run_config
from rag_eval.artifacts.steps.prepare_tests import prepare_tests
from rag_eval.artifacts.steps.read_md_docs import read_md_docs
from rag_eval.artifacts.steps.retrieve import retrieve
from rag_eval.artifacts.steps.split_md_docs_to_chunks import split_md_docs_to_chunks
from rag_eval.artifacts.steps.summarize_results import summarize_results
from rag_eval.config import PipelineConfig
from rag_eval.evaluator.evaluate_test_suite import evaluate_test_suite
from rag_eval.test_runner import create_retrieval_runner


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
        root = ArtifactNode(
            artifact_id="root",
            artifact_type=ArtifactType.ROOT,
            children=[]
        )
        
        print("Reading markdown documents...")
        reader_art = read_md_docs(root=ArtifactRef(artifact_node=root, data=None), config=self._config.md_docs)
        
        print("Splitting markdown documents into chunks...")
        splitter_art = split_md_docs_to_chunks(
            md_docs=reader_art,
            config=self._config.chunking_config,
            runner=self._runner
        )

        print("Embedding document chunks...")
        embeddings_art = embed_doc_chunks(
            doc_chunks=splitter_art,
            config=self._config.docs_embeddings,
            runner=self._runner
        )
        
        print("Generating question collection...")
        question_collection_art = generate_question_collection(
            doc_chunks=splitter_art,
            config=self._config.question_generator,
            runner=self._runner
        )
        
        print("Picking high-quality questions...")
        judged_questions = judge_question_collection(
            question_collection=question_collection_art,
            config=self._config.question_quality_judge,
            runner=self._runner
        )
        filtered_questions_collection = filter_judged_question_collection(
            questions=question_collection_art,
            question_judgements=judged_questions,
            runner=self._runner,
            config=self._config.filter_quality_config,
        )
        
        print("Embedding document chunks with Jina embeddings...")
        jina_embeddings_art = embed_doc_chunks(
            doc_chunks=splitter_art,
            config=LocalEmbedderConfig(
                model=SupportedLocalEmbedder.JINA_EMBEDDINGS_V3
            ),
            runner=self._runner
        )
        
        print("Embedding document chunks with Qwen embeddings...")
        qwen_embeddings_art = embed_doc_chunks(
            doc_chunks=splitter_art,
            config=LocalEmbedderConfig(
                model=SupportedLocalEmbedder.QWEN3_EMBEDDING_4B
            ),
            runner=self._runner
        )
        
        print("Generating candidate chunks...")
        candidate_chunks_art = generate_candidate_chunks(
            question_collection=filtered_questions_collection,
            doc_chunks=splitter_art,
            jina_embeddings=jina_embeddings_art,
            qwen_embeddings=qwen_embeddings_art,
            config=self._config.candidate_answers_generator,
            runner=self._runner
        )
        
        print("Judging candidate chunks...")
        judgement = judge_candidates(
            question_collection=filtered_questions_collection,
            candidate_retrieval=candidate_chunks_art,
            config=self._config.judge,
            runner=self._runner
        )
        
        print("Preparing tests...")
        tests = prepare_tests(
            question_collection=filtered_questions_collection,
            candidate_retrieval_judgements=judgement,
            runner=self._runner,
            config=self._config.test_preparation,
        )
        
        retrievals_runner = create_retrieval_runner(
            config=self._config.runner,
            embedder=create_embedder(self._config.docs_embeddings),
            embedded_doc_chunks=embeddings_art.data.embedded_chunks,
            max_top_k=max(config.retriever_top_k for config in self._config.evaluation.k_configs),
        )       
        retrievals = retrieve(
            test_suite=tests,
            retrieval_runner=retrievals_runner,
            config=self._config.eval_retrieval,
            runner=self._runner
        )
        
        print("Evaluating the pipeline...")
        evaluation = evaluate_test_suite(
            test_suite=tests.data,
            retrievals=retrievals.data,
            configs=self._config.evaluation.k_configs,
        )      
        
        date_now_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        
        print("Summarizing results...")
        summary = summarize_results(
            evaluation=evaluation,
            artifact_id=f"summary_{date_now_str}",
            artifact_store=self._artifact_store
        )
        print(summary)
        persist_run_config(artifact_id=f"run_{date_now_str}", root=root, artifact_store=self._artifact_store)