import photonics_core as pc


def test_detector_counts_photons():
    detector = pc.Detector()
    circuit = pc.Circuit()
    circuit.add_component(detector)

    photons = circuit.run([pc.Photon("a", "H"), pc.Photon("b", "V")])

    assert len(photons) == 2
    assert detector.count() == 2
    detector.reset()
    assert detector.count() == 0
