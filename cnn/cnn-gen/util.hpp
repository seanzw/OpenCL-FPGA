#ifndef UTIL_HPP
#define UTIL_HPP

#include <exception>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <ctime>
#include "include/RapidXML/rapidxml.hpp"

namespace cnn {

    

    std::string fileToString(const std::string &fn) {
        std::string text;
        std::ifstream fs(fn.c_str());
        if (!fs) {
            std::ostringstream os;
            os << "There is no file called " << fn;
            exit(-1);
        }
        text.assign(std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());
        return text;
    }

    

    unsigned int closestMultiple(unsigned int size, unsigned int divisor) {
        unsigned int remainder = size % divisor;
        return remainder == 0 ? size : size - remainder + divisor;
    }

    template <class T>
    void showMatrix(T *matrix, unsigned int width, unsigned int height) {
        for (unsigned int row = 0; row < height; ++row) {
            for (unsigned int col = 0; col < width; ++col) {
                std::cout << matrix[width*row + col] << " ";
            }
            std::cout << std::endl;
        }
        return;
    }

    typedef std::vector<float> vec;
    typedef std::vector<vec> vec2d;

    int getInt(rapidxml::xml_node<> *root, const char *name) {
        rapidxml::xml_node<> *node = root->first_node(name);
        return std::atoi(node->value());
    }

    void getAllItem(rapidxml::xml_node<> *root, vec &items) {
        std::string name = root->name();
        if (name == "item") {
            items.push_back((float)std::atof(root->value()));
        }
        else {
            for (rapidxml::xml_node<> *node = root->first_node(); node; node = node->next_sibling()) {
                getAllItem(node, items);
            }
        }
        return;
    }

    int closestMultiple(int base, int n) {
        int remainder = n % base;
        return remainder == 0 ? n : n - remainder + base;
    }

    void writeXMLOpenTag(std::ofstream &o, const std::string &tag) {
        o << "<" << tag << ">";
    }

    void writeXMLCloseTag(std::ofstream &o, const std::string &tag) {
        o << "</" << tag << ">";
    }

    void writeXMLTag(std::ofstream &o, const std::string &tag, float value) {
        writeXMLOpenTag(o, tag);
        o << value;
        writeXMLCloseTag(o, tag);
        o << std::endl;
    }

    void writeXMLTag(std::ofstream &o, const std::string &tag, size_t value) {
        writeXMLOpenTag(o, tag);
        o << value;
        writeXMLCloseTag(o, tag);
        o << std::endl;
    }

}


#endif
