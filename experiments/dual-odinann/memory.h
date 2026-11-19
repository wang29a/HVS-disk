#pragma once
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

struct Memory {
  unsigned long rss_kib = 0, pss_kib = 0;

  void add(pid_t pid) {
    const auto prefix = "/proc/" + std::to_string(pid) + "/smaps";
    std::ifstream input(prefix + "_rollup");
    if (!input) {
      input.clear();
      input.open(prefix);
    }
    unsigned long rss = 0, pss = 0;
    for (std::string name; std::getline(input, name);) {
      if (name.compare(0, 4, "Rss:") == 0)
        rss += std::stoul(name.substr(4));
      if (name.compare(0, 4, "Pss:") == 0)
        pss += std::stoul(name.substr(4));
    }
    if (!rss || !pss || !input.eof())
      throw std::runtime_error("cannot sample process RSS/PSS");
    rss_kib += rss;
    pss_kib += pss;
  }

  void write(std::ostream &out) const {
    out << "{\"rss_kib\":" << rss_kib << ",\"pss_kib\":" << pss_kib << '}';
  }
};
