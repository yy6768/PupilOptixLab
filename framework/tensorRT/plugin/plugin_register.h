// 
/* 
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "kpn_plugin.h"

#include "tensorRT/logger.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <stack>
#include <unordered_set>
using namespace nvinfer1;

using nvinfer1::plugin::RPROIParams;


namespace {
// This singleton ensures that each plugin is only registered once for a given
// namespace and type, and attempts of duplicate registration are ignored.
class PluginCreatorRegistry {
public:
    static PluginCreatorRegistry &getInstance() {
        static PluginCreatorRegistry instance;
        return instance;
    }

    template<typename CreatorType>
    void addPluginCreator(void *logger, char const *libNamespace) {
        // Make accesses to the plugin creator registry thread safe
        std::lock_guard<std::mutex> lock(mRegistryLock);

        std::string errorMsg;
        std::string verboseMsg;

        std::unique_ptr<CreatorType> pluginCreator{ new CreatorType{} };
        pluginCreator->setPluginNamespace(libNamespace);

        auto logger_ = static_cast<nvinfer1::ILogger *>(logger);
        std::string pluginType = std::string{ pluginCreator->getPluginNamespace() } + "::" + std::string{ pluginCreator->getPluginName() } + " version " + std::string{ pluginCreator->getPluginVersion() };

        if (mRegistryList.find(pluginType) == mRegistryList.end()) {
            bool status = getPluginRegistry()->registerCreator(*pluginCreator, libNamespace);
            if (status) {
                mRegistry.push(std::move(pluginCreator));
                mRegistryList.insert(pluginType);
                verboseMsg = "Registered plugin creator - " + pluginType;
            } else {
                errorMsg = "Could not register plugin creator -  " + pluginType;
            }
        } else {
            verboseMsg = "Plugin creator already registered - " + pluginType;
        }

        if (logger) {
            if (!errorMsg.empty()) {
                logger_->log(ILogger::Severity::kERROR, errorMsg.c_str());
            }
            if (!verboseMsg.empty()) {
                logger_->log(ILogger::Severity::kVERBOSE, verboseMsg.c_str());
            }
        }
    }

    ~PluginCreatorRegistry() {
        std::lock_guard<std::mutex> lock(mRegistryLock);

        // Release pluginCreators in LIFO order of registration.
        while (!mRegistry.empty()) {
            mRegistry.pop();
        }
        mRegistryList.clear();
    }

private:
    PluginCreatorRegistry() {}

    std::mutex mRegistryLock;
    std::stack<std::unique_ptr<IPluginCreator>> mRegistry;
    std::unordered_set<std::string> mRegistryList;

public:
    PluginCreatorRegistry(PluginCreatorRegistry const &) = delete;
    void operator=(PluginCreatorRegistry const &) = delete;
};

template<typename CreatorType>
void initializePlugin(void *logger, char const *libNamespace) {
    PluginCreatorRegistry::getInstance().addPluginCreator<CreatorType>(logger, libNamespace);
}
}// namespace

extern "C" {
bool initLibKPNPlugins(void *logger, const char *libNamespace) {
    initializePlugin<nvinfer1::KPNPluginDynamicCreator>(logger, libNamespace);
    return true;
}
}