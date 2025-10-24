from flask import jsonify, Response
from flask_smorest import Blueprint
from spacey_api.inference.data import load_sentence_data, load_db_pyarrow
from spacey_api.inference.models import load_nq_nasa_v1_pipeline, load_simcse_model, torch_device
from spacey_dev.inference.hybrid_retriever import fused_retriever, retrieve_lexical_ids
from spacey_api.schema.model import SearchRequest, AISearchResponse
from spacey_util.helper import answer_indexes_from_context
import pyarrow as pa, pyarrow.ipc as pa_ipc

df = load_sentence_data()

nq_nasa_v1_pipeline = load_nq_nasa_v1_pipeline()
simcse_model, simcse_tokenizer = load_simcse_model()

model_blueprint = Blueprint("API", __name__, url_prefix="/api", description="model operations")


@model_blueprint.route("/device")
@model_blueprint.response(200)
def device():
    return jsonify({"device": str(torch_device())})


@model_blueprint.route("/health")
@model_blueprint.response(200)
def device():
    return jsonify({"status": "healthy"})


@model_blueprint.route("/data")
@model_blueprint.response(200)
def data():
    records = load_db_pyarrow()

    EXPOSED_COLS = ["id", "title", "content"]
    table = records.select(EXPOSED_COLS)
    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    return Response(
        sink.getvalue().to_pybytes(),
        mimetype="application/vnd.apache.arrow.stream",
        # headers={"Content-Encoding": "gzip"}
    )


@model_blueprint.route("/search", methods=["GET"])
@model_blueprint.arguments(SearchRequest, location="query")
@model_blueprint.response(200, AISearchResponse)
def qa(input):
    question = input["question"]

    print(input['ai_mode'])

    if input['ai_mode']:
        ids, _ = fused_retriever(simcse_model, simcse_tokenizer, question, df)
        text_list = df.iloc[ids]['sentence'].values
        context_ids = df.iloc[ids]['ctx_id'].values
        context = " ".join(text_list)

        out = nq_nasa_v1_pipeline(question=question, context=context, handle_impossible_answer=True)
        answer_text = out['answer'].rstrip()

        ai_answered = answer_text != ""

        matched_ids = []
        if ai_answered:
            answer_indexes = answer_indexes_from_context(out['start'], out['end'], text_list)
            # print(answer_indexes)
            # answer_indexes2 = [i for i, s in enumerate(text_list) if answer_text in s]
            # print(answer_indexes2)
            unique_answer_indexes = set(answer_indexes)
            # context ids non dup
            matched_ids = [context_ids[i] for i in unique_answer_indexes]
            unique_matched_ids = set(matched_ids)
            matched_ids_str = [str(id) for id in unique_matched_ids]

            return jsonify({
                "response": matched_ids_str,
                "ai_answered": ai_answered,
                "ai_text": answer_text
            })
    
    # default fallback bm25
    ids, _ = retrieve_lexical_ids(question, top_k=30)
    contexT_ids = df.iloc[ids]['ctx_id'].values
    non_dup_context_ids = set(contexT_ids)
    str_ids = [str(id) for id in non_dup_context_ids]

    return jsonify({
        "response": str_ids,
        "ai_answered": False
    })