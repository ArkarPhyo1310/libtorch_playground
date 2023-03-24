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
    /**
     * @brief Load Label txt file
     * 
     * @param fileName 
     * @return std::vector<std::string> 
     */
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

    /**
     * @brief Get the Exe Path object
     * 
     * @param exePath 
     * @return fs::path 
     */
    fs::path getExePath(const std::string &exePath)
    {
        return fs::path(exePath);
    }

    /**
     * @brief Get the Default Path of 'Assets' folder
     * 
     * @param exePath 
     * @param path 
     * @return fs::path 
     */
    fs::path getDefault(const fs::path &exePath, const std::string &path)
    {
        fs::path fullDefaultPath = exePath.parent_path() / fs::path("assets") / fs::path(path);
        return fullDefaultPath;
    }

}

#endif
