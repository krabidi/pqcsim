import yaml
import photonics_core as pc

def load_circuit_from_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    circuit = pc.Circuit()
    detectors = []

    for comp in data['circuit']:
        comp_type = comp['type']
        if comp_type == "SPDCSource":
            circuit.add_component(pc.SPDCSource())
        elif comp_type == "Beamsplitter":
            circuit.add_component(pc.Beamsplitter())
        elif comp_type == "Detector":
            d = pc.Detector()
            circuit.add_component(d)
            detectors.append(d)
        else:
            raise ValueError(f"Unknown component type: {comp_type}")

    return circuit, detectors
