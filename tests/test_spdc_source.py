import photonics_core as pc


def test_spdc_source_emits_photon_pairs():
    source = pc.SPDCSource()
    detector = pc.Detector()

    circuit = pc.Circuit()
    circuit.add_component(source)
    circuit.add_component(detector)

    photons = circuit.run()

    assert len(photons) == 2
    assert detector.count() == 2
    modes = {p.mode for p in photons}
    assert any("signal" in m for m in modes)
    assert any("idler" in m for m in modes)
    assert photons[0].pair_id == photons[1].pair_id
