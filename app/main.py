from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from json_input import JsonInput
from model import Model


app = FastAPI()
model = Model('all-MiniLM-L6-v2')


@app.post("/compute_cosine_similarity/")
def compute_cosine_similarity(json_sentences_input: JsonInput):
    """
    API receiving JSON as body with the following format:\n
        {
            “sentence 1”: “this is phrase 1 written in english”,
            “sentence 2”: “this is phrase 2 written in english”
        }
    and retrieving cosine similarity between the 2 sentences as:\n
        {
            “sentence 1”: “this is phrase 1 written in english”,
            “sentence 2”: “this is phrase 2 written in english”,
            “similarity of the two sentences”: “<int>”
        }
    :param json_sentences_input: JSON containing sentences\n
    :return: JSONResponse
    """
    cosine_similarity = model.cosine_similarity(json_sentences_input.sentence1, json_sentences_input.sentence2)
    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"sentence 1": json_sentences_input.sentence1,
                                 "sentence 2": json_sentences_input.sentence2,
                                 "similarity of the two sentences": cosine_similarity}
                        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handler providing more meaningful insights over RequestValidationError,
    happening when provided an incorrect JSON as input.
    :param request: HTTP request
    :param exc: Exception
    :return: JSONResponse
    """
    exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
    content = {'message': 'Input JSON must follow the right format',
               'format': "{'sentence 1': 'str_sentence1', 'sentence 2': 'str_sentence2'}",
               'exc': str(exc_str)}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)
