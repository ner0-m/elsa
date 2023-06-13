
#include "doctest/doctest.h"
#include "DataContainer.h"
#include "elsaDefines.h"
#include "testHelpers.h"
#include "VolumeDescriptor.h"
#include "transforms/FFT.h"

#ifdef ELSA_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include <thrust/complex.h>
#include <thrust/generate.h>
#include <thrust/execution_policy.h>

#include <random>

TEST_SUITE_BEGIN("core");

#ifdef ELSA_CUDA_ENABLED

/* Althoug one might consider it bad style to test
   detail functions instead of the public interface,
   the point of this test is to ensure the equivalence
   of both possible detail implementations, so my hands
   are tied. */
TEST_CASE_TEMPLATE("fft on host and device", data_t, float, double)
{
    GIVEN("Some container")
    {
        auto setup = [&](size_t dim, size_t size) {
            std::random_device r;

            std::default_random_engine e(r());
            std::uniform_real_distribution<data_t> uniform_dist;

            auto shape = elsa::IndexVector_t(dim);
            shape.setConstant(size);

            auto desc = elsa::VolumeDescriptor(shape);

            auto dc = elsa::DataContainer<elsa::complex<data_t>>(desc);
            thrust::generate(thrust::host, dc.begin(), dc.end(), [&]() {
                elsa::complex<data_t> c;
                c.real(uniform_dist(e));
                c.imag(uniform_dist(e));
                return c;
            });
            return dc;
        };

        size_t size[] = {4096, 512, 64};

        for (size_t dims = 1; dims <= 3; dims++) {
            auto dc1 = setup(dims, size[dims - 1]);
            auto dc2 = dc1;

            const auto& desc = dc1.getDataDescriptor();
            const auto& src_shape = desc.getNumberOfCoefficientsPerDimension();
            const auto& src_dims = desc.getNumberOfDimensions();

            WHEN("Using fft (ORTHO)")
            {
                REQUIRE_UNARY(elsa::detail::fftDevice<elsa::complex<data_t>, false>(
                    dc1.storage().data(), src_shape, src_dims, elsa::FFTNorm::ORTHO));
                elsa::detail::fftHost<elsa::complex<data_t>, false>(
                    dc2.storage().data().get(), src_shape, src_dims, elsa::FFTNorm::ORTHO);
                THEN("CPU and GPU implementation are equivalent")
                {
                    for (elsa::index_t i = 0; i < dc1.getSize(); ++i)
                        REQUIRE_UNARY(elsa::checkApproxEq(dc1[i], dc2[i]));
                }
            }

            dc1 = setup(dims, size[dims - 1]);
            dc2 = dc1;

            WHEN("Using ifft (ORTHO)")
            {
                REQUIRE_UNARY(elsa::detail::fftDevice<elsa::complex<data_t>, true>(
                    dc1.storage().data(), src_shape, src_dims, elsa::FFTNorm::ORTHO));
                elsa::detail::fftHost<elsa::complex<data_t>, true>(
                    dc2.storage().data().get(), src_shape, src_dims, elsa::FFTNorm::ORTHO);
                THEN("CPU and GPU implementation are equivalent")
                {
                    for (elsa::index_t i = 0; i < dc1.getSize(); ++i)
                        REQUIRE_UNARY(elsa::checkApproxEq(dc1[i], dc2[i]));
                }
            }

            dc1 = setup(dims, size[dims - 1]);
            dc2 = dc1;

            WHEN("Using fft (FORWARD)")
            {
                REQUIRE_UNARY(elsa::detail::fftDevice<elsa::complex<data_t>, false>(
                    dc1.storage().data(), src_shape, src_dims, elsa::FFTNorm::FORWARD));
                elsa::detail::fftHost<elsa::complex<data_t>, false>(
                    dc2.storage().data().get(), src_shape, src_dims, elsa::FFTNorm::FORWARD);
                THEN("CPU and GPU implementation are equivalent")
                {
                    for (elsa::index_t i = 0; i < dc1.getSize(); ++i)
                        REQUIRE_UNARY(elsa::checkApproxEq(dc1[i], dc2[i]));
                }
            }

            dc1 = setup(dims, size[dims - 1]);
            dc2 = dc1;

            WHEN("Using ifft (BACKWARD)")
            {
                REQUIRE_UNARY(elsa::detail::fftDevice<elsa::complex<data_t>, true>(
                    dc1.storage().data(), src_shape, src_dims, elsa::FFTNorm::BACKWARD));
                elsa::detail::fftHost<elsa::complex<data_t>, true>(
                    dc2.storage().data().get(), src_shape, src_dims, elsa::FFTNorm::BACKWARD);
                THEN("CPU and GPU implementation are equivalent")
                {
                    for (elsa::index_t i = 0; i < dc1.getSize(); ++i)
                        REQUIRE_UNARY(elsa::checkApproxEq(dc1[i], dc2[i]));
                }
            }
        }
    }
}
#endif

TEST_CASE("1D fft in numpy")
{
    using namespace elsa;

    constexpr size_t dim = 10;
    Vector_t<elsa::complex<float>> inputs(dim);
    inputs << complex<float>(0.22270932715095826, 0.46328431255222746),
        complex<float>(0.7728936927702302, 0.3582897386994869),
        complex<float>(0.6298054094517744, 0.9595461883180398),
        complex<float>(0.21851589821484974, 0.2350580044144016),
        complex<float>(0.75002099889163, 0.094482942129406),
        complex<float>(0.5592076821676369, 0.20584398882694932),
        complex<float>(0.1786754307911973, 0.5758593091748346),
        complex<float>(0.6442165368790972, 0.8158957494444451),
        complex<float>(0.5979199081208609, 0.5908647616961652),
        complex<float>(0.615159318836822, 0.1584550825482285);
    /* generated with numpy */
    Vector_t<elsa::complex<float>> expected_ortho(dim);
    expected_ortho << complex<float>(1.6409451543855753, 1.4096105898451798),
        complex<float>(-0.06936164640981084, 0.07384429645133075),
        complex<float>(0.3202121280049301, -0.2985773983765442),
        complex<float>(-0.2899152485001561, -0.3238298117434775),
        complex<float>(-0.3001084639603943, 0.37660368767414043),
        complex<float>(-0.13625054494401148, 0.2879237839882413),
        complex<float>(0.017789365660228526, -0.052393960993486595),
        complex<float>(-0.1994363348894002, 0.2611124729110585),
        complex<float>(-0.44251883875340753, -0.37725817853039695),
        complex<float>(0.16291315936705297, 0.1079981506643282);
    Vector_t<elsa::complex<float>> expected_forward(dim);
    expected_forward << complex<float>(0.5189124203275057, 0.44575800778041846),
        complex<float>(-0.02193407849142431, 0.02335161689988944),
        complex<float>(0.1012599658904968, -0.09441846367173402),
        complex<float>(-0.09167925136742079, -0.10240397793729308),
        complex<float>(-0.09490262912094022, 0.1190925428268964),
        complex<float>(-0.04308620544622152, 0.09104949499371617),
        complex<float>(0.005625491361590718, -0.01656842523774361),
        complex<float>(-0.06306730664466098, 0.08257101398779614),
        complex<float>(-0.1399367437993554, -0.11929951100824883),
        complex<float>(0.05151766444138826, 0.03415201391853039);
    Vector_t<elsa::complex<float>> expected_backward(dim);
    expected_backward << complex<float>(5.189124203275057, 4.4575800778041845),
        complex<float>(-0.21934078491424308, 0.23351616899889438),
        complex<float>(1.012599658904968, -0.9441846367173402),
        complex<float>(-0.9167925136742079, -1.0240397793729308),
        complex<float>(-0.9490262912094021, 1.1909254282689639),
        complex<float>(-0.4308620544622152, 0.9104949499371616),
        complex<float>(0.056254913615907176, -0.16568425237743611),
        complex<float>(-0.6306730664466098, 0.8257101398779613),
        complex<float>(-1.399367437993554, -1.1929951100824883),
        complex<float>(0.5151766444138826, 0.3415201391853039);

    IndexVector_t numCoeff(1);
    numCoeff << dim;

    elsa::VolumeDescriptor desc(numCoeff);

    auto test = [&](elsa::FFTNorm norm, const Vector_t<elsa::complex<float>>& expected) {
        elsa::DataContainer<elsa::complex<float>> dc1(desc, inputs);
        elsa::fft(dc1.storage(), dc1.getDataDescriptor(), norm);
        THEN("Elsa FFT implementation and numpy are equivalent")
        {
            for (elsa::index_t i = 0; i < dc1.getSize(); ++i) {
                REQUIRE_UNARY(elsa::checkApproxEq(dc1[i], expected[i]));
            }
        }
    };

    GIVEN("A random array of inputs")
    {
        WHEN("Using 1D fft (ORTHO)")
        {
            test(elsa::FFTNorm::ORTHO, expected_ortho);
        }

        WHEN("Using 1D fft (FORWARD)")
        {
            test(elsa::FFTNorm::FORWARD, expected_forward);
        }

        WHEN("Using 1D fft (BACKWARD)")
        {
            test(elsa::FFTNorm::BACKWARD, expected_backward);
        }
    }
}

TEST_CASE("2D fft in numpy")
{
    using namespace elsa;

    constexpr size_t rows = 8;
    constexpr size_t cols = 8;
    Vector_t<elsa::complex<float>> inputs(rows * cols);
    inputs << complex<float>(0.22270932715095826, 0.8042800678537125),
        complex<float>(0.7728936927702302, 0.1771606603691005),
        complex<float>(0.6298054094517744, 0.7365852093801976),
        complex<float>(0.21851589821484974, 0.8045300524543826),
        complex<float>(0.75002099889163, 0.9751522388612208),
        complex<float>(0.5592076821676369, 0.02636133534048024),
        complex<float>(0.1786754307911973, 0.09328086042117112),
        complex<float>(0.6442165368790972, 0.6762799452219139),
        complex<float>(0.5979199081208609, 0.8704927439555653),
        complex<float>(0.615159318836822, 0.8944134123564289),
        complex<float>(0.46328431255222746, 0.2229275866775552),
        complex<float>(0.3582897386994869, 0.8787930130900189),
        complex<float>(0.9595461883180398, 0.849769201015244),
        complex<float>(0.2350580044144016, 0.1799706637782298),
        complex<float>(0.094482942129406, 0.4676029035681779),
        complex<float>(0.20584398882694932, 0.5607825069200432),
        complex<float>(0.5758593091748346, 0.6497643410946692),
        complex<float>(0.8158957494444451, 0.6635368664205045),
        complex<float>(0.5908647616961652, 0.5511338669637339),
        complex<float>(0.1584550825482285, 0.3768532270347894),
        complex<float>(0.9295622439017862, 0.016188176642401353),
        complex<float>(0.5431157097465331, 0.6944482268468689),
        complex<float>(0.12452439718288633, 0.8015377120752264),
        complex<float>(0.4512626262108099, 0.8575203188352206),
        complex<float>(0.8312010654818547, 0.8612681155041668),
        complex<float>(0.9218958952435977, 0.4119919116542906),
        complex<float>(0.37058611515495943, 0.4201859291226435),
        complex<float>(0.5751366049957112, 0.40194094986902384),
        complex<float>(0.05778337084375962, 0.18106219555187442),
        complex<float>(0.04476236211779616, 0.5767580264387551),
        complex<float>(0.42807437031266715, 0.8166624121505429),
        complex<float>(0.4250996098912655, 0.20845219080999722),
        complex<float>(0.5766951077470401, 0.2713379314841098),
        complex<float>(0.4546342143273826, 0.6893557674293549),
        complex<float>(0.36670343708107034, 0.10362899932421965),
        complex<float>(0.6808002435560485, 0.22153241731179096),
        complex<float>(0.662804779707341, 0.2321440590746504),
        complex<float>(0.9092882064287654, 0.16036660658461588),
        complex<float>(0.8767788583268561, 0.2615466351308019),
        complex<float>(0.07048017283606467, 0.26363283130375115),
        complex<float>(0.34332589365095145, 0.5140593785593737),
        complex<float>(0.3321056427289044, 0.25205856549558514),
        complex<float>(0.8592441353233319, 0.22277989493999695),
        complex<float>(0.002667256971774945, 0.7658339056362022),
        complex<float>(0.24775289510967302, 0.36946472436896816),
        complex<float>(0.75319810511566, 0.20648703681430125),
        complex<float>(0.04852250598415386, 0.2770362261081264),
        complex<float>(0.1474819524115979, 0.23021322935880306),
        complex<float>(0.08184275977332622, 0.42799726877874),
        complex<float>(0.032711599053945495, 0.05825977787419223),
        complex<float>(0.9973857160215187, 0.6118159334295805),
        complex<float>(0.8364973882518587, 0.1766125637519994),
        complex<float>(0.660356839592562, 0.5311221241105095),
        complex<float>(0.18481096157356458, 0.480149247838938),
        complex<float>(0.38916031772386817, 0.4934746285294136),
        complex<float>(0.4831917252414719, 0.7701267362182836),
        complex<float>(0.37733767587917144, 0.22615601236501315),
        complex<float>(0.5894706386525609, 0.5776011388615859),
        complex<float>(0.4266101714983781, 0.693609844519457),
        complex<float>(0.7726387896809671, 0.5964881344025087),
        complex<float>(0.320648186416873, 0.26890894672030685),
        complex<float>(0.6657961289713672, 0.7544386573113783),
        complex<float>(0.432391871696287, 0.34233011335316665),
        complex<float>(0.5909491745842932, 0.5685678886876933);
    /* generated with numpy */
    Vector_t<elsa::complex<float>> expected_ortho(rows * cols);
    expected_ortho << complex<float>(3.815499000510186, 3.7908530117444452),
        complex<float>(-0.05194618307107265, -0.1751237673362818),
        complex<float>(-0.07956627658223712, -0.10943104394332791),
        complex<float>(-0.06397745670930621, 0.2591342909180308),
        complex<float>(0.05261632516166515, 0.00047355866418797606),
        complex<float>(-0.1911509876244047, -0.057458525509200324),
        complex<float>(0.30913422579071576, 0.3426882365049566),
        complex<float>(-0.18371760049654978, 0.5742200985525387),
        complex<float>(0.34249347117236467, 0.36928514248235694),
        complex<float>(0.2798955858642872, -0.34238719647789845),
        complex<float>(-0.2540389369488166, -0.20849159652653076),
        complex<float>(-0.5236012035231503, -0.08667133945210098),
        complex<float>(-0.1296274987645543, -0.03983143354995161),
        complex<float>(0.11392360434214067, 0.12480256863341635),
        complex<float>(0.5246650839036284, -0.17145165353740535),
        complex<float>(0.07429498389089213, -0.304339564435896),
        complex<float>(0.07187441681379396, -0.012108269584041114),
        complex<float>(0.20693630015390657, 0.13870994014479945),
        complex<float>(0.02153301819704404, 0.1475261755691715),
        complex<float>(0.36616680937385343, -0.2947154473313952),
        complex<float>(-0.09722789331710509, -0.23152273141568),
        complex<float>(-0.40160700499763247, 0.3166092997629574),
        complex<float>(0.5193203777123919, 0.14423872546530325),
        complex<float>(-0.24816566334065615, 0.0564116364425117),
        complex<float>(-0.15560494075483564, 0.10480236921012438),
        complex<float>(0.10691644259336439, -0.12717683287166104),
        complex<float>(0.16259311691955106, 0.10216103263792578),
        complex<float>(-0.45054977276978797, 0.3322145339970006),
        complex<float>(0.2196320339084895, 0.17194402708134576),
        complex<float>(0.06560729624734797, -0.0009365699434099483),
        complex<float>(0.001756152822946537, 0.1291370229531184),
        complex<float>(0.31593006921623873, 0.010913693335674686),
        complex<float>(0.29193279535624966, -0.12642385324681),
        complex<float>(-0.44526225446115114, 0.21363194494210136),
        complex<float>(-0.1433078648587465, -0.00910790083877344),
        complex<float>(-0.4807029416129972, -0.1423901203674324),
        complex<float>(0.14682672607930544, 0.11559230941535474),
        complex<float>(0.09220233250360794, -0.28947330694825485),
        complex<float>(0.06671643448233863, -0.09665811040002077),
        complex<float>(-0.22108326677528597, 0.016232396270947885),
        complex<float>(-0.232672368250224, 0.02234049491397072),
        complex<float>(0.3825820958473371, -0.22481002064457806),
        complex<float>(-0.7123664203366552, -0.0444553296988801),
        complex<float>(0.013608699119954229, 0.38829561855947864),
        complex<float>(-0.3539720414820665, 0.4724936324826853),
        complex<float>(-0.1570982667769417, -0.07127407071291747),
        complex<float>(0.25558227414235174, 0.11283910587919527),
        complex<float>(0.111893521239147, 0.0815068160212441),
        complex<float>(0.10780878548374007, -0.40373308014085807),
        complex<float>(0.34658698925857473, 0.01622068287660675),
        complex<float>(-0.1748907761171683, -0.026056406382395994),
        complex<float>(0.050377473788312185, -0.10730138102326953),
        complex<float>(-0.12513680693996942, 0.34482505609348396),
        complex<float>(0.3162061776896433, -0.23584912394556828),
        complex<float>(-0.35867206077826747, 0.6946729170423901),
        complex<float>(-0.31770541108078704, -0.5522152146451894),
        complex<float>(-0.2652861840139021, 0.5486145545229899),
        complex<float>(-0.060283917632091726, 0.04479478572607702),
        complex<float>(0.0670056235599298, 0.6280523662524555),
        complex<float>(-0.3367345180252318, 0.065663928751785),
        complex<float>(-0.12673348839201898, 0.09099196435899844),
        complex<float>(-0.3715228555002062, -0.5742841070323588),
        complex<float>(0.12303499969022606, 0.2634689328499474),
        complex<float>(-0.44726576369404136, 0.26355566971220906);
    Vector_t<elsa::complex<float>> expected_forward(rows * cols);
    expected_forward << complex<float>(0.4769373750637734, 0.47385662646805576),
        complex<float>(-0.006493272883884082, -0.021890470917035228),
        complex<float>(-0.009945784572779646, -0.01367888049291599),
        complex<float>(-0.007997182088663273, 0.032391786364753865),
        complex<float>(0.006577040645208143, 5.9194833023497664e-05),
        complex<float>(-0.023893873453050586, -0.007182315688650045),
        complex<float>(0.03864177822383948, 0.042836029563119576),
        complex<float>(-0.02296470006206873, 0.07177751231906734),
        complex<float>(0.04281168389654559, 0.046160642810294625),
        complex<float>(0.03498694823303591, -0.04279839955973731),
        complex<float>(-0.03175486711860208, -0.026061449565816353),
        complex<float>(-0.06545015044039379, -0.010833917431512623),
        complex<float>(-0.01620343734556929, -0.004978929193743951),
        complex<float>(0.01424045054276759, 0.015600321079177052),
        complex<float>(0.06558313548795357, -0.021431456692175672),
        complex<float>(0.009286872986361518, -0.038042445554487006),
        complex<float>(0.008984302101724267, -0.0015135336980051095),
        complex<float>(0.025867037519238335, 0.017338742518099935),
        complex<float>(0.002691627274630506, 0.01844077194614644),
        complex<float>(0.045770851171731686, -0.0368394309164244),
        complex<float>(-0.012153486664638138, -0.028940341426960003),
        complex<float>(-0.05020087562470406, 0.03957616247036968),
        complex<float>(0.064915047214049, 0.018029840683162907),
        complex<float>(-0.031020707917582022, 0.00705145455531396),
        complex<float>(-0.019450617594354454, 0.013100296151265567),
        complex<float>(0.013364555324170548, -0.015897104108957637),
        complex<float>(0.02032413961494388, 0.012770129079740705),
        complex<float>(-0.05631872159622352, 0.04152681674962508),
        complex<float>(0.027454004238561196, 0.021493003385168217),
        complex<float>(0.008200912030918505, -0.00011707124292624493),
        complex<float>(0.00021951910286831678, 0.016142127869139804),
        complex<float>(0.03949125865202984, 0.001364211666959339),
        complex<float>(0.0364915994195312, -0.015802981655851256),
        complex<float>(-0.055657781807643913, 0.026703993117762677),
        complex<float>(-0.017913483107343316, -0.0011384876048466802),
        complex<float>(-0.06008786770162465, -0.017798765045929053),
        complex<float>(0.018353340759913184, 0.014449038676919346),
        complex<float>(0.011525291562950997, -0.03618416336853186),
        complex<float>(0.008339554310292341, -0.012082263800002602),
        complex<float>(-0.02763540834691075, 0.002029049533868489),
        complex<float>(-0.02908404603127801, 0.0027925618642463436),
        complex<float>(0.04782276198091715, -0.028101252580572264),
        complex<float>(-0.08904580254208193, -0.0055569162123600135),
        complex<float>(0.00170108738999427, 0.048536952319934844),
        complex<float>(-0.04424650518525832, 0.05906170406033567),
        complex<float>(-0.019637283347117715, -0.008909258839114689),
        complex<float>(0.031947784267793974, 0.014104888234899408),
        complex<float>(0.013986690154893379, 0.010188352002655514),
        complex<float>(0.013476098185467494, -0.05046663501760723),
        complex<float>(0.043323373657321855, 0.002027585359575845),
        complex<float>(-0.021861347014646038, -0.0032570507977995027),
        complex<float>(0.0062971842235390214, -0.013412672627908694),
        complex<float>(-0.01564210086749618, 0.0431031320116855),
        complex<float>(0.039525772211205415, -0.029481140493196036),
        complex<float>(-0.04483400759728343, 0.08683411463029878),
        complex<float>(-0.03971317638509839, -0.06902690183064869),
        complex<float>(-0.033160773001737774, 0.06857681931537375),
        complex<float>(-0.00753548970401147, 0.005599348215759629),
        complex<float>(0.008375702944991223, 0.07850654578155694),
        complex<float>(-0.04209181475315399, 0.008207991093973133),
        complex<float>(-0.015841686049002372, 0.011373995544874806),
        complex<float>(-0.04644035693752578, -0.07178551337904486),
        complex<float>(0.015379374961278263, 0.03293361660624343),
        complex<float>(-0.05590822046175518, 0.032944458714026126);
    Vector_t<elsa::complex<float>> expected_backward(rows * cols);
    expected_backward << complex<float>(30.5239920040815, 30.32682409395557),
        complex<float>(-0.41556946456858124, -1.4009901386902546),
        complex<float>(-0.6365302126578973, -0.8754483515466234),
        complex<float>(-0.5118196536744495, 2.0730743273442473),
        complex<float>(0.42093060129332116, 0.0037884693135038505),
        complex<float>(-1.5292079009952375, -0.45966820407360287),
        complex<float>(2.473073806325727, 2.741505892039653),
        complex<float>(-1.4697408039723987, 4.59376078842031),
        complex<float>(2.739947769378918, 2.954281139858856),
        complex<float>(2.239164686914298, -2.739097571823188),
        complex<float>(-2.032311495590533, -1.6679327722122466),
        complex<float>(-4.1888096281852025, -0.6933707156168079),
        complex<float>(-1.0370199901164345, -0.3186514683996129),
        complex<float>(0.9113888347371257, 0.9984205490673314),
        complex<float>(4.197320671229028, -1.371613228299243),
        complex<float>(0.5943598711271372, -2.4347165154871684),
        complex<float>(0.5749953345103531, -0.09686615667232701),
        complex<float>(1.6554904012312535, 1.1096795211583959),
        complex<float>(0.1722641455763524, 1.1802094045533722),
        complex<float>(2.929334474990828, -2.3577235786511617),
        complex<float>(-0.7778231465368408, -1.8521818513254402),
        complex<float>(-3.2128560399810597, 2.5328743981036594),
        complex<float>(4.154563021699136, 1.153909803722426),
        complex<float>(-1.9853253067252494, 0.45129309154009345),
        complex<float>(-1.244839526038685, 0.8384189536809963),
        complex<float>(0.8553315407469151, -1.0174146629732888),
        complex<float>(1.3007449353564082, 0.8172882611034051),
        complex<float>(-3.604398182158305, 2.657716271976005),
        complex<float>(1.7570562712679165, 1.3755522166507659),
        complex<float>(0.5248583699787843, -0.007492559547279676),
        complex<float>(0.014049222583572274, 1.0330961836249475),
        complex<float>(2.52744055372991, 0.0873095466853977),
        complex<float>(2.335462362849997, -1.0113908259744804),
        complex<float>(-3.5620980356892105, 1.7090555595368113),
        complex<float>(-1.1464629188699722, -0.07286320671018753),
        complex<float>(-3.8456235329039776, -1.1391209629394594),
        complex<float>(1.1746138086344438, 0.9247384753228381),
        complex<float>(0.7376186600288638, -2.3157864555860392),
        complex<float>(0.5337314758587098, -0.7732648832001665),
        complex<float>(-1.768666134202288, 0.1298591701675833),
        complex<float>(-1.8613789460017927, 0.178723959311766),
        complex<float>(3.0606567667786977, -1.798480165156625),
        complex<float>(-5.698931362693243, -0.35564263759104087),
        complex<float>(0.10886959295963328, 3.10636494847583),
        complex<float>(-2.8317763318565325, 3.7799490598614827),
        complex<float>(-1.2567861342155338, -0.5701925657033401),
        complex<float>(2.0446581931388144, 0.9027128470335621),
        complex<float>(0.8951481699131763, 0.6520545281699529),
        complex<float>(0.8624702838699196, -3.229864641126863),
        complex<float>(2.7726959140685987, 0.12976546301285408),
        complex<float>(-1.3991262089373464, -0.20845125105916817),
        complex<float>(0.40301979030649737, -0.8584110481861564),
        complex<float>(-1.0010944555197556, 2.758600448747872),
        complex<float>(2.5296494215171466, -1.8867929915645463),
        complex<float>(-2.8693764862261397, 5.557383336339122),
        complex<float>(-2.5416432886462967, -4.417721717161516),
        complex<float>(-2.1222894721112175, 4.38891643618392),
        complex<float>(-0.4822713410567341, 0.3583582858086163),
        complex<float>(0.5360449884794383, 5.024418930019644),
        complex<float>(-2.6938761442018553, 0.5253114300142805),
        complex<float>(-1.0138679071361518, 0.7279357148719876),
        complex<float>(-2.97218284400165, -4.594272856258871),
        complex<float>(0.9842799975218088, 2.1077514627995795),
        complex<float>(-3.5781261095523313, 2.108445357697672);

    IndexVector_t numCoeff(2);
    numCoeff << rows, cols;

    elsa::VolumeDescriptor desc(numCoeff);

    auto test = [&](elsa::FFTNorm norm, const Vector_t<elsa::complex<float>>& expected) {
        elsa::DataContainer<elsa::complex<float>> dc(desc, inputs);
        elsa::fft(dc.storage(), dc.getDataDescriptor(), norm);
        THEN("Elsa FFT implementation and numpy are equivalent")
        {
            for (elsa::index_t i = 0; i < dc.getSize(); ++i) {
                REQUIRE_UNARY(elsa::checkApproxEq(dc[i], expected[i]));
            }
        }
    };

    GIVEN("A random array of inputs")
    {
        WHEN("Using 2D fft (ORTHO)")
        {
            test(elsa::FFTNorm::ORTHO, expected_ortho);
        }

        WHEN("Using 2D fft (FORWARD)")
        {
            test(elsa::FFTNorm::FORWARD, expected_forward);
        }

        WHEN("Using 2D fft (BACKWARD)")
        {
            test(elsa::FFTNorm::BACKWARD, expected_backward);
        }
    }
}

TEST_SUITE_END();
