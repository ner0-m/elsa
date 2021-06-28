#include "doctest/doctest.h"

#include "Graph.h"
#include <numeric>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("ml");

// TODO(dfrank): remove and replace with proper doctest usage of test cases
#define SECTION(name) DOCTEST_SUBCASE(name)

TEST_CASE("Graph")
{
    SECTION("Graph of int")
    {
        // Construct a graph
        ml::detail::Graph<float> g({{0, 1}, {1, 2}, {0, 2}, {2, 0}});
        REQUIRE(g.getNumberOfNodes() == 3);

        // Adding an edge including a new node increases the overall number of nodes
        g.addEdge(1, 3);
        REQUIRE(g.getNumberOfNodes() == 4);

        // Adding an edge that doesn't introduce a new node doesn't increase the number of nodes
        g.addEdge(0, 1);
        REQUIRE(g.getNumberOfNodes() == 4);

        // Add some data
        std::vector<float> v(static_cast<std::size_t>(g.getNumberOfNodes()));
        std::generate(std::begin(v), std::end(v),
                      []() { return Eigen::internal::random<float>(); });

        for (index_t i = 0; i < g.getNumberOfNodes(); ++i) {
            g.setData(i, &v[std::size_t(i)]);
        }

        for (index_t i = 0; i < g.getNumberOfNodes(); ++i) {
            REQUIRE(*g.getData(i) == Approx(v[std::size_t(i)]));
        }

        // Get all edges from given node
        auto edges = g.getOutgoingEdges(1);
        REQUIRE(edges.size() == 2);
        for (const auto& e : edges) {
            REQUIRE(e.begin()->getIndex() == 1);
            REQUIRE(*e.begin()->getData() == v[1]);
        }

        // Get all edges into a given node
        auto inedges = g.getIncomingEdges(2);
        REQUIRE(inedges.size() == 2);
        for (const auto& e : inedges) {
            REQUIRE(e.end()->getIndex() == 2);
            REQUIRE(*e.end()->getData() == v[2]);
        }

        // Changing the received edge list changes nodes in the graph
        float t = 123.456f;
        edges[0].begin()->setData(&t);

        REQUIRE(g.getData(1) == &t);
        REQUIRE(*g.getData(1) == t);
    }

    SECTION("Graph Visitor")
    {
        ml::detail::Graph<int> g({{0, 1}, {1, 2}, {0, 2}, {2, 0}});
        std::vector<int> v({0, 1, 2});
        std::vector<int> cv(v);
        for (index_t i = 0; i < g.getNumberOfNodes(); ++i) {
            g.setData(i, &v[std::size_t(i)]);
        }
        for (index_t i = 0; i < g.getNumberOfNodes(); ++i) {
            REQUIRE(*g.getData(i) == v[std::size_t(i)]);
        }
        g.visit(0, [](auto node) { *node = *node + 1; });
        for (index_t i = 0; i < g.getNumberOfNodes(); ++i) {
            REQUIRE(*g.getData(i) == v[std::size_t(i)]);
            REQUIRE(*g.getData(i) == cv[std::size_t(i)] + 1);
        }

        int sum = 0;
        g.visit(0, [&sum](auto node) { sum += *node; });
        REQUIRE(sum == std::accumulate(std::begin(v), std::end(v), 0));
    }
    SECTION("Remove edges and nodes")
    {
        ml::detail::Graph<int> g({{0, 1}, {1, 2}, {0, 2}, {2, 0}});

        // There is one outgoing edge from 2, i.e., edge (0,2)
        REQUIRE(g.getOutgoingEdges(2).size() == 1);

        // Add two more
        g.addEdge(2, 1);
        g.addEdge(2, 2);
        REQUIRE(g.getOutgoingEdges(2).size() == 3);

        // Remove all outgoing edges of 2
        g.removeOutgoingEdges(2);
        REQUIRE(g.getOutgoingEdges(2).size() == 0);

        // Add edges again
        g.addEdge(2, 1);
        g.addEdge(2, 2);
        REQUIRE(g.getOutgoingEdges(2).size() == 2);

        // Remove node
        g.removeNode(2);
        REQUIRE(g.getOutgoingEdges(2).size() == 0);
        REQUIRE(g.getIncomingEdges(2).size() == 0);
        REQUIRE(g.getNumberOfNodes() == 2);
    }
    SECTION("Insert node")
    {
        // Build a graph with a central node 2 that is reachable from two nodes
        // 0 and 1 and that has outgoing edges to three nodes 3, 4 and 5
        ml::detail::Graph<int> g({{0, 2}, {1, 2}, {2, 3}, {2, 4}, {2, 5}});
        REQUIRE(g.getOutgoingEdges(2).size() == 3);
        REQUIRE(g.getIncomingEdges(2).size() == 2);

        // Insert a node after 2 with index 6. This node should have exactly one
        // incoming edge (2, 6) and should overtake all outgoing edges of node
        // 2, i.e., it should have the edges (6, 3), (6, 4) and (6, 5).
        // Node 2 should have no other edges than (2, 6) anymore.
        g.insertNodeAfter(2, 6);

        REQUIRE(g.getOutgoingEdges(2).size() == 1);
        REQUIRE(g.getOutgoingEdges(2).front().end()->getIndex() == 6);
        REQUIRE(g.getOutgoingEdges(6).size() == 3);
        REQUIRE(g.getOutgoingEdges(6)[0].end()->getIndex() == 3);
        REQUIRE(g.getOutgoingEdges(6)[1].end()->getIndex() == 4);
        REQUIRE(g.getOutgoingEdges(6)[2].end()->getIndex() == 5);
    }
    SECTION("More visitors")
    {
        ml::detail::Graph<int> g({{0, 1},
                                  {1, 2},
                                  {1, 3},
                                  {2, 4},
                                  {3, 4},
                                  {4, 5},
                                  {5, 6},
                                  {5, 7},
                                  {6, 9},
                                  {7, 8},
                                  {8, 9}});

        std::vector<int> v({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        for (int i = 0; i < 10; ++i)
            g.setData(i, &v[asUnsigned(i)]);

        int sum = 0;

        g.visit(0, [&sum](auto node) { sum += *node; });

        REQUIRE(sum == std::accumulate(v.begin(), v.end(), 0));
    }
}
TEST_SUITE_END();
