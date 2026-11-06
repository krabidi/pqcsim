#include <iostream>
#include <vector>
#include <string>
#include <memory>

namespace pqc {

class Photon {
public:
    std::string mode;
    std::string polarization;
    int pair_id = -1;
    double timestamp = 0.0;

    Photon(std::string m, std::string p, int pid = -1, double time = 0.0)
        : mode(m), polarization(p), pair_id(pid), timestamp(time) {}

    void info() const {
        std::cout << "Photon — Mode: " << mode
                  << ", Pol: " << polarization
                  << ", Pair ID: " << pair_id
                  << ", Time: " << timestamp << std::endl;
    }
};

class Component {
public:
    virtual void process(std::vector<Photon>& photons) = 0;
    virtual std::string name() const = 0;
    virtual ~Component() = default;
};

class Beamsplitter : public Component {
public:
    void process(std::vector<Photon>& photons) override {
        for (auto& photon : photons) {
            photon.mode += "_bsplit";
        }
    }

    std::string name() const override {
        return "Beamsplitter";
    }
};

class Detector : public Component {
private:
    int count_ = 0;

public:
    void process(std::vector<Photon>& photons) override {
        for (const auto& photon : photons) {
            std::cout << "Detected photon in mode: " << photon.mode
                      << ", Pair ID: " << photon.pair_id
                      << ", Time: " << photon.timestamp << std::endl;
            count_++;
        }
        // Don't clear the photon list
    }

    std::string name() const override {
        return "Detector";
    }

    int count() const {
        return count_;
    }

    void reset() {
        count_ = 0;
    }
};

class SPDCSource : public Component {
private:
    int counter = 0;

public:
    void process(std::vector<Photon>& photons) override {
        int id = counter++;
        double time = id * 1.0;

        photons.emplace_back("spdc_" + std::to_string(id) + "_signal", "H", id, time);
        photons.emplace_back("spdc_" + std::to_string(id) + "_idler", "V", id, time);
    }

    std::string name() const override {
        return "SPDCSource";
    }
};

class Circuit {
private:
    std::vector<std::shared_ptr<Component>> components;

public:
    void add_component(std::shared_ptr<Component> c) {
        components.push_back(c);
    }

    // Run every component in order, starting from an optional list of input photons.
    std::vector<Photon> run(std::vector<Photon> photons = {}) {
        for (auto& comp : components) {
            std::cout << "Processing with: " << comp->name() << std::endl;
            comp->process(photons);
        }
        return photons;
    }
};

}  // namespace pqc
