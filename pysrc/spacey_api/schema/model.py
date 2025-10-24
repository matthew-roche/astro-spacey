from marshmallow import Schema, fields, validate

class Record(Schema):
    id = fields.String(required=True)
    title = fields.String()
    body = fields.String()

class SearchRequest(Schema):
    question = fields.String(required=True, validate=validate.Length(min=1))
    ai_mode = fields.Boolean(load_default=False)

class DataResponse(Schema):
    response = fields.List(fields.Nested(Record(partial=True)))

class AISearchResponse(Schema):
    response = fields.List(fields.Nested(Record(partial=True)))
    ai_answered = fields.Boolean(load_default=False)
    ai_text = fields.String(load_default="")
    ctx_id = fields.List(fields.String(), load_default=[])