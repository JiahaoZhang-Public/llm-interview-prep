# P042 Function Schema Parsing — Solution Notes

## Core Approach

1. Parse OpenAI-style function/tool definitions from JSON into a structured format
2. Extract: function name, description, and parameter schema (types, required fields, defaults)
3. Input format follows the OpenAI function calling API: `{"name": ..., "description": ..., "parameters": {...}}`
4. Convert to an internal representation usable for validation, documentation, or prompt construction
5. Handle JSON Schema types: string, integer, number, boolean, array, object, and enum constraints

## Interview Oral Template

> "Function schema parsing converts OpenAI-style tool definitions into a
> structured internal format. Each function has a name, description, and
> a parameters object following JSON Schema. I parse out each parameter's
> name, type, description, whether it's required, and any constraints like
> enums or default values. This parsed schema is used for two things:
> validating the LLM's tool call arguments before execution, and formatting
> tool descriptions in the system prompt. Clean parsing prevents runtime
> errors from malformed arguments."

## Common Pitfalls

- **Missing required fields**: The schema may not always include all expected fields -- must handle optional/missing fields gracefully with defaults
- **Nested parameters**: Parameters can be objects with their own properties -- must recursively parse nested schemas
- **Type coercion**: LLM might return "42" (string) instead of 42 (integer) -- the parser should validate and optionally coerce types
- **Enum validation**: If a parameter has an enum constraint, the provided value must be one of the allowed values

## Complexity

- Time: O(P) where P is the total number of parameters across all functions
- Space: O(F * P) for storing parsed schemas for F functions
- Parsing is a one-time setup cost -- negligible compared to LLM calls
