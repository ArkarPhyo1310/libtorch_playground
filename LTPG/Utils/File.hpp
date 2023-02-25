#ifndef FILE_H
#define FILE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

namespace libtorchPG
{

    std::vector<std::string> loadLabels(const std::string &fileName)
    {
        std::ifstream ins(fileName);
        if (!ins.is_open())
        {
            std::cerr << "Couldn't open " << fileName << std::endl;
            abort();
        }

        std::vector<std::string> labels;
        std::string line;

        while (getline(ins, line))
        {
            labels.push_back(line);
        }
        ins.close();

        return labels;
    }

    fs::path getExePath(const std::string &exePath)
    {
        return fs::path(exePath);
    }

    fs::path getDefault(const fs::path &exePath, const std::string &path)
    {
        fs::path fullDefaultPath = exePath.parent_path() / fs::path("data") / fs::path(path);
        return fullDefaultPath;
    }

}

#endif
