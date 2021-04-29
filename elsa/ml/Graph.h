#pragma once

#include <map>
#include <vector>
#include <fstream>
#include <algorithm>
#include <deque>
#include <memory>
#include <string>
#include <utility>

#include "elsaDefines.h"
#include "Common.h"

namespace elsa::ml
{
    namespace detail
    {
        /// A node in a graph.
        ///
        /// \author David Tellenbach
        ///
        /// \tparam T type of the data this node holds
        /// \tparam UseRawPtr If this parameter is set to `true` data is stored
        /// as a raw pointer, i.e., `T*`. If this parameter is set to `false`
        /// data is stored as `std::shared_ptr<T>`.
        template <typename T, bool UseRawPtr = true>
        class Node
        {
        public:
            using PointerType = std::conditional_t<UseRawPtr, T*, std::shared_ptr<T>>;

            /// construct a node by specifying its index in the graph
            explicit Node(index_t index) : index_(index), data_(nullptr) {}

            /// return the index of this node
            inline index_t getIndex() const { return index_; }

            /// return a pointer to the data held by this node
            PointerType getData() { return data_; }

            /// return a constant pointer to the data held by this node
            const PointerType getData() const { return data_; }

            /// set data held by this node
            inline void setData(PointerType data) { data_ = data; }

        private:
            /// this node's index in the graph
            index_t index_;

            /// the data this node holds
            PointerType data_;
        };

        /// An edge between nodes of the graph
        ///
        /// \author David Tellenbach
        ///
        /// \tparam T type of the data this node holds
        /// \tparam UseRawPtr If this parameter is set to `true` data is stored
        /// as a raw pointer, i.e., `T*`. If this parameter is set to `false`
        /// data is stored as `std::shared_ptr<T>`.
        template <typename T, bool UseRawPtr = true>
        class Edge
        {
        public:
            /// the type of nodes this edge connects
            using NodeType = Node<T, UseRawPtr>;

            /// construct an Edge by specifying its begin and end
            Edge(NodeType* begin, NodeType* end) : begin_(begin), end_(end) {}

            /// get a constant pointer to the begin-node of this edge
            const inline NodeType* begin() const { return begin_; }

            /// get a pointer to the begin-node of this edge
            inline NodeType* begin() { return begin_; }

            /// get a constant pointer to the end-node of this edge
            const inline NodeType* end() const { return end_; }

            /// get a pointer to the end-node of this edge
            inline NodeType* end() { return end_; }

        private:
            /// the begin-node of this edge
            NodeType* begin_;

            /// the end-node of this edge
            NodeType* end_;
        };

        /// A graph that can be altered and traversed in an efficient and
        /// structured manner.
        ///
        /// \author David Tellenbach
        ///
        /// \tparam T type of the data this node holds
        /// \tparam UseRawPtr If this parameter is set to `true` data is stored
        /// as a raw pointer, i.e., `T*`. If this parameter is set to `false`
        /// data is stored as `std::shared_ptr<T>`.
        template <typename T, bool UseRawPtr = true>
        class Graph
        {
        public:
            using PointerType = std::conditional_t<UseRawPtr, T*, std::shared_ptr<T>>;

            /// type of the graph nodes
            using NodeType = Node<T, UseRawPtr>;

            /// type of edges of the graph
            using EdgeType = Edge<T, UseRawPtr>;

            /// default constructor
            Graph() = default;

            /// construct a Graph by specifying an adjacency list, i.e., a list
            /// of edges connecting nodes
            Graph(std::initializer_list<std::pair<index_t, index_t>> edges)
            {
                for (const auto& edge : edges)
                    addEdge(edge.first, edge.second);
            }

            /// return the number of nodes
            inline index_t getNumberOfNodes() const { return static_cast<index_t>(nodes_.size()); }

            /// return a pointer to the data held by the node at index
            inline PointerType getData(index_t index) { return nodes_.at(index).getData(); }

            /// return a constant pointer to the data held by the node at index
            const inline PointerType getData(index_t index) const
            {
                return nodes_.at(index).getData();
            }

            /// set data of node index
            inline void setData(index_t index, PointerType data)
            {
                return nodes_.at(index).setData(data);
            }

            /// add edge from begin to end
            inline void addEdge(index_t begin, index_t end)
            {
                nodes_.insert({begin, NodeType(begin)});
                nodes_.insert({end, NodeType(end)});
                edges_.emplace_back(EdgeType(&nodes_.at(begin), &nodes_.at(end)));
            }

            /// \returns a reference to a vector of edges
            inline std::vector<EdgeType>& getEdges() { return edges_; }

            /// \returns a constant reference to a vector of edges
            inline const std::vector<EdgeType>& getEdges() const { return edges_; }

            /// \returns a vector containing all edges beginning at a given
            /// index
            inline std::vector<EdgeType> getOutgoingEdges(index_t begin) const
            {
                std::vector<EdgeType> ret;
                for (const auto& edge : edges_)
                    if (edge.begin()->getIndex() == begin)
                        ret.push_back(edge);

                return ret;
            }

            /// \returns a vector containing all edges ending at a given index
            inline std::vector<EdgeType> getIncomingEdges(index_t end) const
            {
                std::vector<EdgeType> ret;
                for (const auto& edge : edges_)
                    if (edge.end()->getIndex() == end)
                        ret.push_back(edge);

                return ret;
            }

            /// Insert a new node after a node with a given index.
            ///
            /// \param index Index of the node after that the insertion
            /// should be done
            /// \param newIndex Index of the to be inserted node
            void insertNodeAfter(index_t index, index_t newIndex)
            {
                for (auto& e : getOutgoingEdges(index))
                    addEdge(newIndex, e.end()->getIndex());
                removeOutgoingEdges(index);
                addEdge(index, newIndex);
            }

            /// \returns a reference to a map of indices and nodes
            inline std::map<index_t, NodeType>& getNodes() { return nodes_; }

            /// \returns a constant reference to a map of indices and nodes
            inline const std::map<index_t, NodeType>& getNodes() const { return nodes_; }

            /// delete all nodes and edges of the graph
            inline void clear()
            {
                edges_.clear();
                nodes_.clear();
            }

            /// Remove all outgoing edges from node with index begin
            inline void removeOutgoingEdges(index_t begin)
            {
                edges_.erase(std::remove_if(std::begin(edges_), std::end(edges_),
                                            [&begin](auto& edge) {
                                                return edge.begin()->getIndex() == begin;
                                            }),
                             std::end(edges_));
            }

            /// Remove all incoming edges from node with index end
            inline void removeIncomingEdges(index_t end)
            {
                edges_.erase(
                    std::remove_if(std::begin(edges_), std::end(edges_),
                                   [&end](auto& edge) { return edge.end()->getIndex() == end; }),
                    std::end(edges_));
            }

            /// Remove a node from the graph.
            ///
            /// If preserveConnectivity is true, all begin-nodes of incoming
            /// edges get begin-nodes of end-nodes of outgoing edges.
            inline void removeNode(index_t idx, bool preserveConnectivity = true)
            {
                if (preserveConnectivity) {
                    // Add an edge that bypassed the node that is to be removed
                    auto incomingEdges = getIncomingEdges(idx);
                    auto outgoingEdges = getOutgoingEdges(idx);
                    for (const auto& inEdge : incomingEdges) {
                        auto beginIdx = inEdge.begin()->getIndex();
                        for (const auto& outEdge : outgoingEdges) {
                            auto endIdx = outEdge.end()->getIndex();
                            addEdge(beginIdx, endIdx);
                        }
                    }
                }
                // Remove all edges of the node
                removeOutgoingEdges(idx);
                removeIncomingEdges(idx);

                // Remove the node itself
                nodes_.erase(idx);
            }

            /// Visit all nodes of the graph. The visitors have access the node
            /// indices
            ///
            /// \param root Index of the node that serves as starting point
            /// of the traversal.
            ///
            /// \param visitor A function-like object (function, overloaded
            /// call operator, lambda) with the signature
            ///   `void(T* data, index_t index)`.
            /// The visitor will be applied to every node in a breadth-first
            /// traversal. The semantics of the parameter is
            ///
            /// - `data` is the data held by the current node in the traversal
            /// - `index` is the index of the current node
            template <typename Visitor, typename NextVisitor, typename StopFunctor>
            void visitWithIndex(index_t root, Visitor visitor, NextVisitor nextVisitor,
                                StopFunctor stop)
            {
                visitImpl<Visitor, NextVisitor, StopFunctor, /* access index */ true,
                          /* forward */ true>(root, std::forward<Visitor>(visitor),
                                              std::forward<NextVisitor>(nextVisitor),
                                              std::forward<StopFunctor>(stop));
            }

            template <typename Visitor, typename NextVisitor, typename StopFunctor>
            void visit(index_t root, Visitor visitor, NextVisitor nextVisitor, StopFunctor stop)
            {
                visitImpl<Visitor, NextVisitor, StopFunctor, /* access index */ false,
                          /* forward */ true>(root, std::forward<Visitor>(visitor),
                                              std::forward<NextVisitor>(nextVisitor),
                                              std::forward<StopFunctor>(stop));
            }

            template <typename Visitor, typename NextVisitor>
            void visit(index_t root, Visitor visitor, NextVisitor nextVisitor)
            {
                visit(root, visitor, nextVisitor, []([[maybe_unused]] auto node) { return false; });
            }

            template <typename Visitor>
            void visit(index_t root, Visitor visitor)
            {
                visit(
                    root, visitor,
                    []([[maybe_unused]] auto node, [[maybe_unused]] auto nextNode) {},
                    []([[maybe_unused]] auto node) { return false; });
            }

            template <typename Visitor, typename NextVisitor, typename StopFunctor>
            void visitBackward(index_t root, Visitor visitor, NextVisitor nextVisitor,
                               StopFunctor stop)
            {
                visitImpl<Visitor, NextVisitor, StopFunctor, /* access index */ false,
                          /* forward */ false>(root, std::forward<Visitor>(visitor),
                                               std::forward<NextVisitor>(nextVisitor),
                                               std::forward<StopFunctor>(stop));
            }

            template <typename Visitor, typename NextVisitor, typename StopFunctor>
            void visitBackwardWithIndex(index_t root, Visitor visitor, NextVisitor nextVisitor,
                                        StopFunctor stop)
            {
                visitImpl<Visitor, NextVisitor, StopFunctor, /* access index */ true,
                          /* forward */ false>(root, std::forward<Visitor>(visitor),
                                               std::forward<NextVisitor>(nextVisitor),
                                               std::forward<StopFunctor>(stop));
            }

            enum class RankDir { TD, LR };

            /// Print a representation of the graph in the Dot language.
            ///
            /// \param filename the name of the file that is written to
            /// \param nodePrinter a function (or lambda) that defines how
            ///                        to print the data of a single node
            /// \param dpi Dots-per-inch definition for the Dot representation
            /// \param rankDir RankDir::TD if the graph should be drawn top-down,
            ///                    rankdDir::LR if it should be drawn left-right
            template <typename NodePrinter>
            void toDot(const std::string& filename, NodePrinter nodePrinter, index_t dpi = 90,
                       RankDir rankDir = RankDir::TD)
            {
                std::ofstream os(filename);
                os << "digraph {\n";
                os << "graph [" << (rankDir == RankDir::TD ? "rankdir=TD" : "rankdir=LR")
                   << ", dpi=" << dpi << "];\n";

                // print all nodes
                for (const auto& node : getNodes()) {
                    if (node.second.getData()) {
                        os << nodePrinter(node.second.getData(), node.first) << "\n";
                    } else {
                        os << node.first << "\n";
                    }
                }

                // print all edges
                for (const auto& edge : getEdges())
                    os << edge.begin()->getIndex() << "->" << edge.end()->getIndex() << ";\n";

                os << "}";
            }

        private:
            template <typename Visitor, typename NextVisitor, typename StopFunctor,
                      bool AccessIndex, bool Forward>
            void visitImpl(index_t root, Visitor visitor, NextVisitor nextVisitor, StopFunctor stop)
            {
                if (nodes_.find(root) == nodes_.end())
                    throw std::invalid_argument("Unknown node index");

                // We maintain a list of nodes we've already visited and a call-queue to
                // ensure the correct order of traversal.
                std::vector<bool> visited(asIndex(getNumberOfNodes()));
                // Note that this queue is in fact a deque, so we can push and pop from
                // both, the front and back.
                std::deque<index_t> queue;

                // Perform an iterative depth-first traversal through the graph
                // Push the input-node onto the call-queue
                queue.push_back(root);

                while (!queue.empty()) {
                    index_t s = queue.back();
                    queue.pop_back();

                    if (!visited[static_cast<std::size_t>(s)]) {

                        bool needStop;
                        if constexpr (AccessIndex) {
                            needStop = stop(nodes_.at(s).getData(), s);
                        } else {
                            needStop = stop(nodes_.at(s).getData());
                        }
                        if (!needStop) {
                            if constexpr (AccessIndex) {
                                visitor(nodes_.at(s).getData(), s);
                            } else {
                                visitor(nodes_.at(s).getData());
                            }
                            visited[asIndex(s)] = true;
                        } else {
                            queue.push_front(s);
                            continue;
                        }
                    }

                    if constexpr (Forward) {
                        for (auto& e : getOutgoingEdges(s)) {
                            auto idx = e.end()->getIndex();

                            if (!visited[static_cast<std::size_t>(idx)]) {
                                queue.push_back(idx);
                                if constexpr (AccessIndex) {
                                    nextVisitor(nodes_.at(s).getData(), s, e.end()->getData(), idx);
                                } else {
                                    nextVisitor(nodes_.at(s).getData(), e.end()->getData());
                                }
                            }
                        }
                    } else {
                        for (auto& e : getIncomingEdges(s)) {
                            auto idx = e.begin()->getIndex();

                            if (!visited[static_cast<std::size_t>(idx)]) {
                                queue.push_back(idx);
                                if constexpr (AccessIndex) {
                                    nextVisitor(nodes_.at(s).getData(), s, e.begin()->getData(),
                                                idx);
                                } else {
                                    nextVisitor(nodes_.at(s).getData(), e.begin()->getData());
                                }
                            }
                        }
                    }
                }
            }

            std::vector<EdgeType> edges_;
            std::map<index_t, NodeType> nodes_;
        };

    } // namespace detail
} // namespace elsa::ml