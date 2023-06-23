#include "SymmetrizedDerivative.h"
#include "IdenticalBlocksDescriptor.h"
#include "Timer.h"
#include "FiniteDifferences.h"
#include "Identity.h"
#include "Scaling.h"
#include "FiniteDifferences.h"
#include "BlockLinearOperator.h"

namespace elsa
{

    using namespace std;

    template <typename data_t>
    SymmetrizedDerivative<data_t>::SymmetrizedDerivative(const DataDescriptor& domainDescriptor)
    : LinearOperator<data_t>(
            domainDescriptor,
            IdenticalBlocksDescriptor{
                2 * downcast_safe<BlockDescriptor>(domainDescriptor.clone())->getNumberOfBlocks(),
                downcast_safe<BlockDescriptor>(domainDescriptor.clone())->getDescriptorOfBlock(0)}
            )
    {

        const auto blocksDesc = downcast_safe<BlockDescriptor>(this->getDomainDescriptor().clone());
        std::unique_ptr<DataDescriptor> blockDesc = blocksDesc->getDescriptorOfBlock(0).clone();

        BooleanVector_t bv1(blockDesc->getNumberOfDimensions());
        BooleanVector_t bv2(blockDesc->getNumberOfDimensions());

        bv1[0] = true;
        FiniteDifferences<data_t> D1(*blockDesc, bv1);

        bv2[1] = true;
        FiniteDifferences<data_t> D2(*blockDesc, bv2);

        std::vector<std::unique_ptr<LinearOperator<data_t>>> D1S0_list;
        D1S0_list.emplace_back(D1.clone());
        D1S0_list.emplace_back(Scaling<data_t>(*blockDesc, 0).clone());
        auto D1S0 = BlockLinearOperator<data_t>(
            D1S0_list, BlockLinearOperator<data_t>::BlockType::COL
        );

        std::vector<std::unique_ptr<LinearOperator<data_t>>> D2D1_list;
        D2D1_list.emplace_back(D2.clone());
        D2D1_list.emplace_back(D1.clone());
        auto D2D1 = BlockLinearOperator<data_t>(
            D2D1_list, BlockLinearOperator<data_t>::BlockType::COL
        );

        std::vector<std::unique_ptr<LinearOperator<data_t>>> S0D2_list;
        S0D2_list.emplace_back(Scaling<data_t>(*blockDesc, 0).clone());
        S0D2_list.emplace_back(D2.clone());
        auto S0D2 = BlockLinearOperator<data_t>(
            S0D2_list, BlockLinearOperator<data_t>::BlockType::COL
        );
/*
        cout<<"D1 domain: "<<endl;
        for(auto e : D1.getDomainDescriptor().getNumberOfCoefficientsPerDimension()) cout<<e<<" ";
        cout<<endl;

        cout<<"D1 range: "<<endl;
        for(auto e : D1.getRangeDescriptor().getNumberOfCoefficientsPerDimension()) cout<<e<<" ";
        cout<<endl;

        cout<<"D2D1 domain: "<<endl;
        for(auto e : D2D1.getDomainDescriptor().getNumberOfCoefficientsPerDimension()) cout<<e<<" ";
        cout<<endl;

        cout<<"D2D1 ramge: "<<endl;
        for(auto e : D2D1.getRangeDescriptor().getNumberOfCoefficientsPerDimension()) cout<<e<<" ";
        cout<<endl;
*/
        std::vector<std::unique_ptr<LinearOperator<data_t>>> core_list;
        core_list.emplace_back(D1S0.clone());
        core_list.emplace_back(D2D1.clone());
        core_list.emplace_back(D2D1.clone());
        core_list.emplace_back(S0D2.clone());
        core_ = BlockLinearOperator<data_t>(
            core_list, BlockLinearOperator<data_t>::BlockType::ROW
        ).clone();

        std::vector<std::unique_ptr<LinearOperator<data_t>>> row1_list;
        row1_list.emplace_back(Identity<data_t>(*blockDesc).clone());
        row1_list.emplace_back(Scaling<data_t>(*blockDesc, 0).clone());
        row1_list.emplace_back(Scaling<data_t>(*blockDesc, 0).clone());
        row1_list.emplace_back(Scaling<data_t>(*blockDesc, 0).clone());
        auto row1 = BlockLinearOperator<data_t>(
            row1_list, BlockLinearOperator<data_t>::BlockType::COL
        );

        std::vector<std::unique_ptr<LinearOperator<data_t>>> row23_list;
        row23_list.emplace_back(Scaling<data_t>(*blockDesc, 0).clone());
        row23_list.emplace_back(Scaling<data_t>(*blockDesc, 0.5).clone());
        row23_list.emplace_back(Scaling<data_t>(*blockDesc, 0).clone());
        row23_list.emplace_back(Scaling<data_t>(*blockDesc, 0).clone());
        auto row23 = BlockLinearOperator<data_t>(
            row23_list, BlockLinearOperator<data_t>::BlockType::COL
        );

        std::vector<std::unique_ptr<LinearOperator<data_t>>> row4_list;
        row4_list.emplace_back(Scaling<data_t>(*blockDesc, 0).clone());
        row4_list.emplace_back(Scaling<data_t>(*blockDesc, 0).clone());
        row4_list.emplace_back(Scaling<data_t>(*blockDesc, 0).clone());
        row4_list.emplace_back(Identity<data_t>(*blockDesc).clone());
        auto row4 = BlockLinearOperator<data_t>(
            row4_list, BlockLinearOperator<data_t>::BlockType::COL
        );

        std::vector<std::unique_ptr<LinearOperator<data_t>>> scaling_list;
        scaling_list.emplace_back(row1.clone());
        scaling_list.emplace_back(row23.clone());
        scaling_list.emplace_back(row23.clone());
        scaling_list.emplace_back(row4.clone());
        scaling_ = BlockLinearOperator<data_t>(
                    scaling_list, BlockLinearOperator<data_t>::BlockType::ROW
                    ).clone();

    }

    using namespace std;

    template <typename data_t>
    void SymmetrizedDerivative<data_t>::applyImpl(const DataContainer<data_t>& x1,
                                              DataContainer<data_t>& Ax) const
    {
        Timer timeguard("SymmetrizedDerivative", "apply");

        auto x = DataContainer<data_t>(this->getDomainDescriptor(), ContiguousStorage<data_t>{x1.begin(), x1.end()});

        Ax = scaling_->apply((core_->apply(x)));

        //Ax = (core_->apply(x));

        /*
        const auto blockDesc = downcast_safe<BlockDescriptor>(x.getDataDescriptor().clone());
        if (!blockDesc)
            throw LogicError("SymmetrizedDerivative: cannot get block from not-blocked container");

        Ax = 0;

        BooleanVector_t bv1(blockDesc->getDescriptorOfBlock(0).getNumberOfDimensions());
        BooleanVector_t bv2(blockDesc->getDescriptorOfBlock(0).getNumberOfDimensions());

        bv1[0] = true;
        FiniteDifferences<data_t> D1(blockDesc->getDescriptorOfBlock(0), bv1);

        bv2[1] = true;
        FiniteDifferences<data_t> D2(blockDesc->getDescriptorOfBlock(0), bv2);

        auto block_one = D1.apply(x.getBlock(0));
        auto block_two = (D2.apply(x.getBlock(0)) + D1.apply(x.getBlock(1))) / 2;
        auto block_three = block_two;
        auto block_four = D2.apply(x.getBlock(1));

        auto storage = block_one.storage();
        storage.insert(storage.end(), block_two.begin(), block_two.end());
        storage.insert(storage.end(), block_three.begin(), block_three.end());
        storage.insert(storage.end(), block_four.begin(), block_four.end());

        Ax = DataContainer<data_t>(Ax.getDataDescriptor(), storage);
         */
    }

    template <typename data_t>
    void SymmetrizedDerivative<data_t>::applyAdjointImpl(const DataContainer<data_t>& y,
                                     DataContainer<data_t>& Aty) const
    {
        //cout<<"!!! try apply adjoint !!!"<<endl;

        auto x = DataContainer<data_t>(this->getRangeDescriptor(), ContiguousStorage<data_t>{y.begin(), y.end()});

        Aty = core_->applyAdjoint(scaling_->applyAdjoint(x));
        //Aty = core_->applyAdjoint(x);
        /*
        cout<<"x: "<<x.getNumberOfBlocks()<<endl;
        for(auto el : x) cout<<el<<" ";
        cout<<endl;

        cout<<"b0: "<<endl;
        for(auto el : x.getBlock(0)) cout<<el<<" ";
        cout<<endl;

        cout<<"b1: "<<endl;
        for(auto el : x.getBlock(1)) cout<<el<<" ";
        cout<<endl;

        cout<<"b2: "<<endl;
        for(auto el : x.getBlock(2)) cout<<el<<" ";
        cout<<endl;

        cout<<"b3: "<<endl;
        for(auto el : x.getBlock(3)) cout<<el<<" ";
        cout<<endl;

        const auto blockDesc = downcast_safe<BlockDescriptor>(x.getDataDescriptor().clone());
        if (!blockDesc)
            throw LogicError("SymmetrizedDerivative: cannot get block from not-blocked container");

        BooleanVector_t bv1(blockDesc->getDescriptorOfBlock(0).getNumberOfDimensions());
        BooleanVector_t bv2(blockDesc->getDescriptorOfBlock(0).getNumberOfDimensions());

        bv1[0] = true;
        FiniteDifferences<data_t> D1(blockDesc->getDescriptorOfBlock(0), bv1);

        bv2[1] = true;
        FiniteDifferences<data_t> D2(blockDesc->getDescriptorOfBlock(0), bv2);

        auto block_one = D1.applyAdjoint(x.getBlock(0));
        auto block_two = (D2.applyAdjoint(x.getBlock(0)) + D1.applyAdjoint(x.getBlock(1))) / 2;
        auto block_three = block_two;
        auto block_four = D2.applyAdjoint(x.getBlock(1));

        auto block_one_two = block_one + block_two;
        auto block_three_four = block_three + block_four;

        auto storage = block_one_two.storage();
        storage.insert(storage.end(), block_three_four.begin(), block_three_four.end());



        Aty = DataContainer<data_t>(this->getDomainDescriptor());
*/
    }

    template <typename data_t>
    SymmetrizedDerivative<data_t>* SymmetrizedDerivative<data_t>::cloneImpl() const
    {
        return new SymmetrizedDerivative(this->getDomainDescriptor());
    }

    template <typename data_t>
    bool SymmetrizedDerivative<data_t>::isEqual(const LinearOperator<data_t>& other) const
    {
        return true;
    }

    // ------------------------------------------
    // explicit template instantiation
    template class SymmetrizedDerivative<float>;
    template class SymmetrizedDerivative<double>;
    template class SymmetrizedDerivative<complex<float>>;
    template class SymmetrizedDerivative<complex<double>>;

}  // namespace elsa