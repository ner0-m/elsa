#pragma once

#include <string>
#include <fstream>

#include "DataContainer.h"
#include "VolumeDescriptor.h"
#include "State.h"

namespace elsa::ml
{
    /// Common ml utilities
    ///
    /// @author David Tellenbach
    struct Utils {
        /// Utilities to plot a model
        struct Plotting {

            /// Direction for plotting the model graph.
            enum class RankDir {
                /// Plot model from top to bottom
                TD,
                /// Plot model from left to right
                LR
            };

            /// Convert a model to Graphviz' DOT format.
            ///
            /// @param model The model to plot
            /// @param filename The filename of the DOT file. This parameter is
            /// optional and defaults to "model.png"
            /// @param rankDir The direction of the plotted DOT graph. This
            /// parameter is optional and defaults to RankDir::TD.
            /// @param dpi The dots-per-inch encoded in the DOT file.
            template <typename T>
            static void modelToDot([[maybe_unused]] const T& model,
                                   const std::string& filename = "model.png",
                                   RankDir rankDir = RankDir::TD, int dpi = 96)
            {
                std::ofstream os(filename);

                auto& graph = detail::State<real_t>::getGraph();

                // Use Eigen's formatting facilities to control the format of in- and output shapes
                Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ", ", "", "", "(", ")");

                graph.toDot(
                    filename,
                    [&fmt](auto layer, index_t idx) {
                        std::stringstream ss;
                        ss << idx << " [shape=\"record\", label=\"" << layer->getName() << ": "
                           << detail::getEnumMemberAsString(layer->getLayerType())
                           << " | {input: \\l| output:\\l} | { [";
                        for (int i = 0; i < layer->getNumberOfInputs(); ++i) {
                            ss << layer->getInputDescriptor(i)
                                      .getNumberOfCoefficientsPerDimension()
                                      .format(fmt)
                               << (i == layer->getNumberOfInputs() - 1 ? "]" : ", ");
                        }
                        ss << " \\l | "
                           << layer->getOutputDescriptor()
                                  .getNumberOfCoefficientsPerDimension()
                                  .format(fmt)
                           << "\\l}\"];\n";
                        return ss.str();
                    },
                    dpi,
                    rankDir == RankDir::TD ? std::decay_t<decltype(graph)>::RankDir::TD
                                           : std::decay_t<decltype(graph)>::RankDir::LR);
            }
        };

        /// Common encoding routines
        struct Encoding {
            /// Encode a DataContainer in one-hot encoding
            ///
            /// We expect the input-data to be shaped as ([1,] batchSize). The
            /// new one-hot encoded descriptor has shape (num_classes, batch_size)
            template <typename data_t>
            static DataContainer<data_t> toOneHot(const DataContainer<data_t>& dc,
                                                  index_t numClasses, index_t batchSize)
            {
                Eigen::VectorXf oneHotData(batchSize * numClasses);
                oneHotData.setZero();

                index_t segmentIdx = 0;
                for (index_t i = 0; i < batchSize; ++i) {
                    oneHotData.segment(segmentIdx, numClasses)
                        .coeffRef(static_cast<index_t>(dc[i])) = data_t(1);
                    segmentIdx += numClasses;
                }

                IndexVector_t dims{{numClasses, batchSize}};
                return DataContainer<data_t>(VolumeDescriptor(dims), oneHotData);
            }

            /// Decode a DataContainer that is encoded in one-hot encoding.
            template <typename data_t>
            static DataContainer<data_t> fromOneHot(const DataContainer<data_t> dc,
                                                    index_t numClasses)
            {
                index_t batchSize = dc.getDataDescriptor().getNumberOfCoefficientsPerDimension()(1);
                Eigen::VectorXf data(batchSize);
                data.setZero();
#ifndef ELSA_CUDA_VECTOR
                auto expr = (data_t(1) * dc).eval();
#else
                Eigen::VectorXf expr(dc.getSize());
                for (index_t i = 0; i < dc.getSize(); ++i) {
                    expr[i] = dc[i];
                }
#endif
                for (int i = 0; i < batchSize; ++i) {
                    index_t maxIdx;
                    expr.segment(i * numClasses, numClasses).maxCoeff(&maxIdx);
                    data[i] = static_cast<data_t>(maxIdx);
                }
                IndexVector_t dims{{batchSize}};
                VolumeDescriptor desc(dims);
                return DataContainer<data_t>(desc, data);
            }
        };
    };
} // namespace elsa::ml