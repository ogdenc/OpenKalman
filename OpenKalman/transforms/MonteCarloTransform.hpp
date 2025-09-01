/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of MonteCarloTransform.
 */

#ifndef OPENKALMAN_MONTECARLOTRANSFORM_HPP
#define OPENKALMAN_MONTECARLOTRANSFORM_HPP

#include <cmath>
#include <execution>
#include <iterator>
#include <numeric>

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  /**
   * \brief A Monte Carlo transform from one Gaussian distribution to another.
   * \details Uses ideas from Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1979),
   * "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances."
   * Technical Report STAN-CS-79-773, Department of Computer Science, Stanford University.
   * http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
   */
  struct MonteCarloTransform : oin::TransformBase<MonteCarloTransform>
  {
  private:

    using Base = oin::TransformBase<MonteCarloTransform>;


    template<bool return_cross, typename TransformationType, typename InputDistribution, typename...NoiseDistributions>
    struct MonteCarloSet
    {
    private:

      using In_Mean = typename DistributionTraits<InputDistribution>::Mean;
      using Out_Mean = std::invoke_result_t<TransformationType, In_Mean>;
      using InputCoefficients = vector_space_descriptor_of_t<In_Mean, 0>;
      using OutputCoefficients = vector_space_descriptor_of_t<Out_Mean, 0>;
      using Scalar = scalar_type_of_t<In_Mean>;
      static_assert((coordinates::compares_with<OutputCoefficients,
        typename DistributionTraits<NoiseDistributions>::StaticDescriptor> and ...));

      using InputMeanMatrix = dense_writable_matrix_t<In_Mean, data_layout::none, Scalar,
        std::tuple<coordinates::dimension_of<InputCoefficients>, Axis>>;
      using OutputEuclideanMeanMatrix = dense_writable_matrix_t<InputMeanMatrix, data_layout::none, Scalar, std::tuple<coordinates::dimension_of<OutputCoefficients>, Axis>>;
      using OutputCovarianceMatrix = dense_writable_matrix_t<InputMeanMatrix, data_layout::none, Scalar,
        std::tuple<coordinates::dimension_of<OutputCoefficients>, coordinates::dimension_of<OutputCoefficients>>>;
      using OutputCovarianceSA = typename MatrixTraits<std::decay_t<OutputCovarianceMatrix>>::template SelfAdjointMatrixFrom<>;
      using CrossCovarianceMatrix = dense_writable_matrix_t<InputMeanMatrix, data_layout::none, Scalar,
        std::tuple<coordinates::dimension_of<InputCoefficients>, coordinates::dimension_of<OutputCoefficients>>>;

      using InputMean = Mean<InputCoefficients, InputMeanMatrix>;
      using OutputEuclideanMean = EuclideanMean<OutputCoefficients, OutputEuclideanMeanMatrix>;
      using OutputCovariance = Covariance<OutputCoefficients, OutputCovarianceSA>;
      using CrossCovariance = Matrix<InputCoefficients, OutputCoefficients, CrossCovarianceMatrix>;


      struct MonteCarloSum0
      {
        std::size_t count;
        InputMean x;
        OutputEuclideanMean y_E;
        OutputCovariance yy;
      };


      struct MonteCarloSum1
      {
        std::size_t count;
        InputMean x;
        OutputEuclideanMean y_E;
        OutputCovariance yy;
        CrossCovariance xy;
      };

    public:

      using MonteCarloSum = std::conditional_t<return_cross, MonteCarloSum1, MonteCarloSum0>;


      static constexpr auto
      zero()
      {
        if constexpr (return_cross)
          return MonteCarloSum {
            0, make_zero<InputMean>(), make_zero<OutputEuclideanMean>(),
            make_zero<OutputCovariance>(), make_zero<CrossCovariance>()};
        else
          return MonteCarloSum {
            0, make_zero<InputMean>(), make_zero<OutputEuclideanMean>(),
            make_zero<OutputCovariance>()};
      }


      static auto
      one(const TransformationType& trans, const InputDistribution& dist, const NoiseDistributions&...noise)
      {
        const auto x = dist();
        const auto y = trans(x, noise()...);
        if constexpr (return_cross)
          return MonteCarloSum {1, x, to_euclidean(y), make_zero<OutputCovariance>(),
            make_zero<CrossCovariance>()};
        else
          return MonteCarloSum {1, x, to_euclidean(y), make_zero<OutputCovariance>()};
      }


      struct MonteCarloBinaryOp
      {
        auto operator()(const MonteCarloSum& set1, const MonteCarloSum& set2)
        {
          using Scalar = scalar_type_of_t<decltype(set1.y_E)>;
          if (set2.count == 1) // This is most likely. Take advantage of prior knowledge of set2.
          {
            const auto count = set1.count + 1;
            const Scalar s_count = count;
            const auto x = (set1.count * set1.x + set2.x) / s_count;
            const auto y_E = (set1.count * set1.y_E + set2.y_E) / s_count;
            const auto delta = from_euclidean(set2.y_E) - from_euclidean(set1.y_E);
            const auto delta_adj_factor = adjoint(delta) * set1.count / s_count;
            const OutputCovariance yy {set1.yy + delta * delta_adj_factor};
            if constexpr (return_cross)
            {
              const auto xy = set1.xy + (set2.x - set1.x) * delta_adj_factor;
              return MonteCarloSum {count, x, y_E, yy, xy};
            }
            else
            {
              return MonteCarloSum {count, x, y_E, yy};
            }
          }
          else
          {
            const auto count = set1.count + set2.count;
            const Scalar s_count = count;
            const auto x = (set1.count * set1.x + set2.count * set2.x) / s_count;
            const auto y_E = (set1.count * set1.y_E + set2.count * set2.y_E) / s_count;
            const auto delta = from_euclidean(set2.y_E) - from_euclidean(set1.y_E);
            const auto delta_adj_factor = adjoint(delta) * set1.count * set2.count / s_count;
            const OutputCovariance yy {set1.yy + set2.yy + delta * delta_adj_factor};
            if constexpr (return_cross)
            {
              const auto xy = set1.xy + set2.xy + (set2.x - set1.x) * delta_adj_factor;
              return MonteCarloSum {count, x, y_E, yy, xy};
            }
            else
            {
              return MonteCarloSum {count, x, y_E, yy};
            }
          }
        }
      };


      struct iterator
      {
        iterator(const std::function<MonteCarloSum()>& g, std::size_t initial_position = 0)
          : one_sum_generator(g), position(initial_position) {}

        using iterator_category = std::forward_iterator_tag;
        using value_type = MonteCarloSum;
        using reference = MonteCarloSum&;
        using pointer = MonteCarloSum*;
        using difference_type = std::ptrdiff_t;

        auto& operator=(const iterator& other) { if (this != &other) position = other.position; }

        auto operator*() const { return one_sum_generator(); }

        auto& operator++() { ++position; return *this; }
        auto operator++(int) { const auto temp = *this; ++position; return temp; }

        auto operator==(const iterator& other) const { return position == other.position; }
        auto operator!=(const iterator& other) const { return position != other.position; }

      private:

        const std::function<MonteCarloSum()>& one_sum_generator;

        std::size_t position;

      };


      MonteCarloSet(const TransformationType& trans, std::size_t s, const InputDistribution& dist,
                    const NoiseDistributions&...noise)
        : samples(s),
          one_sum_generator([&trans, &dist, &noise...] { return one(trans, dist, noise...); })
      {}


      auto begin() { return iterator(one_sum_generator, 0); }


      auto end() { return iterator(one_sum_generator, samples); }

    private:

      const std::size_t samples;

      const std::function<MonteCarloSum()> one_sum_generator;

    };

  public:

    /**
     * \brief Constructor
     * \param samples The number of random samples taken from the prior distribution.
     */
    explicit MonteCarloTransform(const std::size_t samples = 100000) : size {samples} {}


    using Base::operator();


    /**
     * \brief Perform a Monte Carlo transform from one statistical distribution to another.
     * \tparam InputDist The prior distribution.
     * \tparam Trans The tests on which the transform is based.
     * \tparam NoiseDists Zero or more noise distributions.
     * \return The posterior distribution.
     **/
#ifdef __cpp_concepts
    template<gaussian_distribution InputDist, linearized_function<1> Trans, gaussian_distribution ... NoiseDists>
    requires requires(Trans g, InputDist x, NoiseDists...n) { g(mean_of(x), mean_of(n)...); }
#else
    template<typename InputDist, typename Trans, typename ... NoiseDists, std::enable_if_t<
      (gaussian_distribution<InputDist> and ... and gaussian_distribution<NoiseDists>) and
      linearized_function<Trans, 1> and std::is_invocable_v<Trans, typename DistributionTraits<InputDist>::Mean,
        typename DistributionTraits<NoiseDists>::Mean...>, int> = 0>
#endif
    auto operator()(const InputDist& x, const Trans& transformation, const NoiseDists& ...n) const
    {
      using MSet = MonteCarloSet<false, Trans, InputDist, NoiseDists...>;
      auto m_set = MSet(transformation, size, x, n...);
      auto binary_op = typename MSet::MonteCarloBinaryOp();
      using MSum = typename MSet::MonteCarloSum;

      MSum m_sum = std::reduce(std::execution::par_unseq, m_set.begin(), m_set.end(), MSet::zero(), binary_op);
      auto mean_output = make_self_contained(from_euclidean(m_sum.y_E));
      auto out_covariance = make_self_contained(m_sum.yy / (size - 1.));
      return GaussianDistribution {mean_output, out_covariance};
    }


    using Base::transform_with_cross_covariance;


    /**
     * \brief Perform a Monte Carlo transform, also returning the cross-covariance.
     * \tparam InputDist The prior distribution.
     * \tparam Trans The tests on which the transform is based.
     * \tparam NoiseDists Zero or more noise distributions.
     * \return A tuple comprising the posterior distribution and the cross-covariance.
     **/
#ifdef __cpp_concepts
    template<gaussian_distribution InputDist, linearized_function<1> Trans, gaussian_distribution ... NoiseDists>
    requires requires(Trans g, InputDist x, NoiseDists...n) { g(mean_of(x), mean_of(n)...); }
#else
    template<typename InputDist, typename Trans, typename ... NoiseDists, std::enable_if_t<
      (gaussian_distribution<InputDist> and ... and gaussian_distribution<NoiseDists>) and
      linearized_function<Trans, 1> and std::is_invocable_v<Trans, typename DistributionTraits<InputDist>::Mean,
        typename DistributionTraits<NoiseDists>::Mean...>, int> = 0>
#endif
    auto transform_with_cross_covariance(const InputDist& x, const Trans& transformation, const NoiseDists& ...n) const
    {
      using MSet = MonteCarloSet<true, Trans, InputDist, NoiseDists...>;
      auto m_set = MSet(transformation, size, x, n...);
      auto binary_op = typename MSet::MonteCarloBinaryOp();
      using MSum = typename MSet::MonteCarloSum;

      MSum m_sum = std::reduce(std::execution::par_unseq, m_set.begin(), m_set.end(), MSet::zero(), binary_op);
      auto mean_output = make_self_contained(from_euclidean(m_sum.y_E));
      auto out_covariance = make_self_contained(m_sum.yy / (size - 1.));
      auto cross_covariance = make_self_contained(m_sum.xy / (size - 1.));
      auto out = GaussianDistribution {mean_output, out_covariance};
      return std::tuple {std::move(out), std::move(cross_covariance)};
    }

  private:

    const std::size_t size;

  };


}


#endif
