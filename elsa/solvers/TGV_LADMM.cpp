#include "LinearizedADMM.h"
#include "LinearOperator.h"
#include "ProximalOperator.h"
#include "ProximalL2Squared.h"
#include "ProximalL1.h"
#include "ProximalBoxConstraint.h"
#include "TGV_LADMM.h"
#include "PowerIterations.h"
#include "elsaDefines.h"
#include "DataContainer.h"
#include "Logger.h"
#include "FiniteDifferences.h"
#include "BlockLinearOperator.h"
#include "CombinedProximal.h"
#include "Identity.h"
#include "EmptyTransform.h"
#include "Scaling.h"
#include "SymmetrizedDerivative.h"
#include "IdenticalBlocksDescriptor.h"

namespace elsa
{

    using namespace std;

    template <class data_t>
    TGV_LADMM<data_t>::TGV_LADMM(const LinearOperator<data_t>& A, const DataContainer<data_t>& b)
    : A_(A.clone()), b_(b)
    {}

    template <class data_t>
    DataContainer<data_t> TGV_LADMM<data_t>::solve(index_t iterations,
                                                        std::optional<DataContainer<data_t>> x0)
    {

        auto D = FiniteDifferences<data_t>(A_->getDomainDescriptor());

        std::vector<std::unique_ptr<LinearOperator<data_t>>> l_row1;
        l_row1.emplace_back(A_->clone());
        l_row1.emplace_back(make_unique<EmptyTransform<data_t>>(D.getRangeDescriptor(), A_->getRangeDescriptor()));
        BlockLinearOperator<data_t> row1(l_row1, BlockLinearOperator<data_t>::BlockType::COL);

        std::vector<std::unique_ptr<LinearOperator<data_t>>> l_row2;
        l_row2.emplace_back(std::make_unique<FiniteDifferences<data_t>>(A_->getDomainDescriptor()));
        l_row2.emplace_back(std::make_unique<Scaling<data_t>>(D.getRangeDescriptor(), -1)); //negative ID
        BlockLinearOperator<data_t> row2(l_row2, BlockLinearOperator<data_t>::BlockType::COL);

        std::vector<std::unique_ptr<LinearOperator<data_t>>> l_mod_scale;
        l_mod_scale.emplace_back(std::make_unique<Scaling<data_t>>(A_->getDomainDescriptor(), 0));
        l_mod_scale.emplace_back(std::make_unique<Scaling<data_t>>(A_->getDomainDescriptor(), 0));
        l_mod_scale.emplace_back(std::make_unique<Scaling<data_t>>(A_->getDomainDescriptor(), 0));
        l_mod_scale.emplace_back(std::make_unique<Scaling<data_t>>(A_->getDomainDescriptor(), 0));
        BlockLinearOperator<data_t> mod_scale(l_mod_scale, BlockLinearOperator<data_t>::BlockType::ROW);

        std::vector<std::unique_ptr<LinearOperator<data_t>>> l_row3;
        l_row3.emplace_back(mod_scale.clone());
        l_row3.emplace_back(std::make_unique<SymmetrizedDerivative<data_t>>(D.getRangeDescriptor()));
        BlockLinearOperator<data_t> row3(l_row3, BlockLinearOperator<data_t>::BlockType::COL);

        std::vector<std::unique_ptr<LinearOperator<data_t>>> K_ops;
        K_ops.emplace_back(row1.clone());
        K_ops.emplace_back(row2.clone());
        K_ops.emplace_back(row3.clone());

        auto K = BlockLinearOperator<data_t>(
                K_ops, BlockLinearOperator<data_t>::BlockType::ROW
            );

        auto proxg1 = ProximalL2Squared<data_t>(b_); //добавить нулей
        auto proxg2 = ProximalL1<data_t>();
        auto proxg3 = ProximalL1<data_t>();

        auto proxg = CombinedProximal<data_t>(proxg1, proxg2, proxg3);
        auto proxf = ProximalBoxConstraint<data_t>(0);

        /*auto Knorm = powerIterations(adjoint(K) * K);
        cout<<"K norm: "<<endl;
        for(auto el : K.getRangeDescriptor().getNumberOfCoefficientsPerDimension()) cout<<el<<" ";
        cout<<endl;

        cout<<"K range: "<<endl;
        for(auto el : K.getRangeDescriptor().getNumberOfCoefficientsPerDimension()) cout<<el<<" ";
        cout<<endl;*/


        /*
        auto range = K.apply(b_);
        auto domain = K.applyAdjoint(range);
        std::cout<<"adj TGV: "<<std::endl;
        for(auto el : domain) std::cout<<el<<" ";

        auto v = DataContainer<data_t>(K.getDomainDescriptor());
        int count = 0;
        for(auto& el : v) el = count++;
        auto T_v = K.apply(v);
        auto u = DataContainer<data_t>(K.getRangeDescriptor());
        for(auto& el : u) el = count--;
        auto Tj_u = K.applyAdjoint(u);
        auto Tvu = T_v * u;
        auto uTjv = v * Tj_u;
        std::cout<<"sum1: "<<Tvu.sum()<<std::endl;
        std::cout<<"sum2: "<<uTjv.sum()<<std::endl;
        std::cout<<"Tvu: "<<std::endl;
        //for(auto el : Tvu) std::cout<<el<<" ";
        std::cout<<std::endl;
        std::cout<<"uTjv: "<<std::endl;
        //for(auto el : uTjv) std::cout<<el<<" ";
        std::cout<<std::endl;
*/
        cout<<"K range: "<<endl;
        for(auto el : K.getRangeDescriptor().getNumberOfCoefficientsPerDimension()) cout<<el<<" ";
        cout<<endl;

        cout<<"K domain: "<<endl;
        for(auto el : K.getDomainDescriptor().getNumberOfCoefficientsPerDimension()) cout<<el<<" ";
        cout<<endl;

        //proxf - easy
        //proxg - hard

        auto admm = LinearizedADMM<data_t>(K, proxf, proxg, 10, 0.0006);
        auto res = admm.solve(iterations);
        cout<<"dims: "<<res.getNumberOfBlocks()<<endl;
        cout<<"size: "<<res.getBlock(0).getDataDescriptor().getNumberOfCoefficients()<<endl;
        auto final = DataContainer<data_t>(A_->getDomainDescriptor(),
                                           ContiguousStorage<data_t>(
                                               res.getBlock(0).begin(), res.getBlock(0).end()
                                               ));

        return final;

    }

    template <class data_t>
    TGV_LADMM<data_t>* TGV_LADMM<data_t>::cloneImpl() const
    {
        return new TGV_LADMM<data_t>(*A_, b_);
    }

    template <class data_t>
    bool TGV_LADMM<data_t>::isEqual(const Solver<data_t>& other) const
    {
        return false;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class TGV_LADMM<float>;
    template class TGV_LADMM<double>;
} // namespace elsa
