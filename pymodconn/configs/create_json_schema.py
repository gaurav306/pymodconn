import ruamel.yaml
import json

yaml = ruamel.yaml.YAML()

def create_schema_from_yaml(yaml_path, schema_path):
	yaml_configs = ruamel.yaml.YAML().load(open(yaml_path))

	schema = {"$schema": "http://json-schema.org/draft-07/schema#",
			  "type": "object",
			  "required": [],
			  "properties": {}}

	for k, v in yaml_configs.items():
		if isinstance(v, str):
			schema["properties"][k] = {"type": "string"}
		elif isinstance(v, int):
			schema["properties"][k] = {"type": "integer", "minimum": v, "maximum": v}
		elif isinstance(v, float):
			schema["properties"][k] = {"type": "number", "minimum": v, "maximum": v}
		elif isinstance(v, bool):
			schema["properties"][k] = {"type": "boolean"}
		elif isinstance(v, list):
			schema["properties"][k] = {"type": "array"}
			if len(v) > 0:
				schema["properties"][k]["items"] = create_schema_from_list(v)
			
		elif isinstance(v, dict):
			schema["properties"][k] = {"type": "object"}
			schema["properties"][k]["properties"] = create_schema_from_dict(v)
			
	schema['required'] = list(yaml_configs.keys())

	with open(schema_path, "w") as schema_file:
		json.dump(schema, schema_file, indent=4)

	return schema


def create_schema_from_list(lst):
	schema = {}
	if isinstance(lst[0], str):
		schema = {"type": "string"}
	elif isinstance(lst[0], int):
		schema = {"type": "integer"}
	elif isinstance(lst[0], float):
		schema = {"type": "number"}
	elif isinstance(lst[0], bool):
		schema = {"type": "boolean"}
	elif isinstance(lst[0], list):
		schema = {"type": "array"}
		if len(lst[0]) > 0:
			schema["items"] = create_schema_from_list(lst[0])
	elif isinstance(lst[0], dict):
		schema = {"type": "object"}
		schema["properties"] = create_schema_from_dict(lst[0])
		
	return schema


def create_schema_from_dict(dct):
	schema = {}
	for k, v in dct.items():
		if isinstance(v, str):
			schema[k] = {"type": "string"}
		elif isinstance(v, int):
			schema[k] = {"type": "integer"}
		elif isinstance(v, float):
			schema[k] = {"type": "number"}
		elif isinstance(v, bool):
			schema[k] = {"type": "boolean"}
		elif isinstance(v, list):
			schema[k] = {"type": "array"}
			if len(v) > 0:
				schema[k]["items"] = create_schema_from_list(v)
		elif isinstance(v, dict):
			schema[k] = {"type": "object"}
			schema[k]["properties"] = create_schema_from_dict(v)
	
	
	return schema

