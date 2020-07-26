//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_PARAMETERS_H
#define EFANNA2E_PARAMETERS_H

#include <sstream>
#include <typeinfo>
#include <unordered_map>
namespace efanna2e {

class Parameters {
 public:
  template <typename ParamType>
  inline void Set(const std::string &name, const ParamType &value) {
    std::stringstream sstream;
    sstream << value;
    params[name] = sstream.str();
  }

  // NOTE: pybind11 do not support template functions.
  // Since add auto type convertion is not trival,
  // we just return string directly as a work around.
  inline std::string GetRaw(const std::string &name) const {
    auto item = params.find(name);
    if (item == params.end()) {
      throw std::invalid_argument("Invalid parameter name.");
    } else {
      return item->second;
    }
  }

  template <typename ParamType>
  inline ParamType Get(const std::string &name) const {
    auto item = params.find(name);
    if (item == params.end()) {
      throw std::invalid_argument("Invalid parameter name.");
    } else {
      return ConvertStrToValue<ParamType>(item->second);
    }
  }

  template <typename ParamType>
  inline ParamType Get(const std::string &name,
                       const ParamType &default_value) {
    try {
      return Get<ParamType>(name);
    } catch (std::invalid_argument e) {
      return default_value;
    }
  }

 private:
  std::unordered_map<std::string, std::string> params;

  template <typename ParamType>
  inline ParamType ConvertStrToValue(const std::string &str) const {
    std::stringstream sstream(str);
    ParamType value;
    if (!(sstream >> value) || !sstream.eof()) {
      std::stringstream err;
      err << "Failed to convert value '" << str
          << "' to type: " << typeid(value).name();
      throw std::runtime_error(err.str());
    }
    return value;
  }
};

}  // namespace efanna2e

#endif  // EFANNA2E_PARAMETERS_H
