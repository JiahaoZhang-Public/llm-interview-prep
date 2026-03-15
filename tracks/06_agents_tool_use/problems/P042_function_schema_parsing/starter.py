def parse_function_schema(schema):
    params = schema.get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])

    return {
        "name": schema["name"],
        "parameters": {
            name: prop.get("type", "string")
            for name, prop in properties.items()
        },
        "required": required,
    }
