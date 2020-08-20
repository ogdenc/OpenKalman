/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_MONTECARLOTRANSFORM_H
#define OPENKALMAN_MONTECARLOTRANSFORM_H

#include <cmath>
#include <execution>
#include <iterator>
#include <numeric>

namespace OpenKalman
{
  /**
   * @brief A Monte Carlo transform from one Gaussian distribution to another.
   * Uses ideas from Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1979),
   * "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances."
   * Technical Report STAN-CS-79-773, Department of Computer Science, Stanford University.
   * http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
   */
  template<
    typename TransformationType> ///< The transformation on which the transform is based.
  struct MonteCarloTransform;


  namespace internal
  {
    template<typename TransformationType, typename InputDistribution, typename...NoiseDistributions>
    struct MonteCarloSet
    {
    protected:
      using InputCoefficients = typename TransformationType::InputCoefficients;
      using OutputCoefficients = typename TransformationType::OutputCoefficients;
      static_assert(is_equivalent_v<InputCoefficients, typename DistributionTraits<InputDistribution>::Coefficients>);
      static_assert(std::conjunction_v<is_equivalent<OutputCoefficients,
        typename DistributionTraits<NoiseDistributions>::Coefficients>...>);

      using InputMeanMatrix = typename MatrixTraits<
        typename DistributionTraits<InputDistribution>::Mean>::template StrictMatrix<InputCoefficients::size, 1>;
      using OutputEuclideanMeanMatrix =
        typename MatrixTraits<InputMeanMatrix>::template StrictMatrix<OutputCoefficients::size, 1>;
      using OutputCovarianceMatrix =
        typename MatrixTraits<InputMeanMatrix>::template StrictMatrix<OutputCoefficients::size, OutputCoefficients::size>;
      using OutputCovarianceSA = typename MatrixTraits<OutputCovarianceMatrix>::template SelfAdjointBaseType<>;
      using CrossCovarianceMatrix =
        typename MatrixTraits<InputMeanMatrix>::template StrictMatrix<InputCoefficients::size, OutputCoefficients::size>;

      using InputMean = Mean<InputCoefficients, InputMeanMatrix>;
      using OutputEuclideanMean = EuclideanMean<OutputCoefficients, OutputEuclideanMeanMatrix>;
      using OutputCovariance = Covariance<OutputCoefficients, OutputCovarianceSA>;
      using CrossCovariance = TypedMatrix<InputCoefficients, OutputCoefficients, CrossCovarianceMatrix>;

    public:
      struct MonteCarloSum
      {
        std::size_t count;
        InputMean x;
        OutputEuclideanMean y_E;
        OutputCovariance yy;
        CrossCovariance xy;
      };

      static constexpr auto
      zero()
      {
        return MonteCarloSum {
          0, MatrixTraits<InputMean>::zero(), MatrixTraits<OutputEuclideanMean>::zero(),
          MatrixTraits<OutputCovariance>::zero(), MatrixTraits<CrossCovariance>::zero()};
      }

      static auto
      one(const TransformationType& trans, const InputDistribution& dist, const NoiseDistributions&...noise)
      {
        const auto x = dist();
        const auto y = trans(x, noise()...);
        return MonteCarloSum {1, x, to_Euclidean(y), OutputCovariance::zero(), CrossCovariance::zero()};
      }

      struct MonteCarloBinaryOp
      {
        auto operator()(const MonteCarloSum& set1, const MonteCarloSum& set2)
        {
          using Scalar = typename MatrixTraits<decltype(set1.y_E)>::Scalar;
          if (set2.count == 1) // This is most likely. Take advantage of prior knowledge of set2.
          {
            const auto count = set1.count + 1;
            const Scalar s_count = count;
            const auto x = (set1.count * set1.x + set2.x) / s_count;
            const auto y_E = (set1.count * set1.y_E + set2.y_E) / s_count;
            const auto delta = from_Euclidean(set2.y_E) - from_Euclidean(set1.y_E);
            const auto delta_adj_factor = adjoint(delta) * set1.count / s_count;
            const auto yy = set1.yy + delta * delta_adj_factor;
            const auto xy = set1.xy + (set2.x - set1.x) * delta_adj_factor;
            return MonteCarloSum {count, x, y_E, yy, xy};
          }
          else
          {
            const auto count = set1.count + set2.count;
            const Scalar s_count = count;
            const auto x = (set1.count * set1.x + set2.count * set2.x) / s_count;
            const auto y_E = (set1.count * set1.y_E + set2.count * set2.y_E) / s_count;
            const auto delta = from_Euclidean(set2.y_E) - from_Euclidean(set1.y_E);
            const auto delta_adj_factor = adjoint(delta) * set1.count * set2.count / s_count;
            const auto yy = set1.yy + set2.yy + delta * delta_adj_factor;
            const auto xy = set1.xy + set2.xy + (set2.x - set1.x) * delta_adj_factor;
            return MonteCarloSum {count, x, y_E, yy, xy};
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

      protected:
        const std::function<MonteCarloSum()>& one_sum_generator;
        std::size_t position;
      };

      MonteCarloSet(
        const TransformationType& trans, std::size_t s, const InputDistribution& dist, const NoiseDistributions&...noise)
        : samples(s),
          one_sum_generator([trans, dist, noise...] { return one(trans, dist, noise...); })
      {}

      auto begin() { return iterator(one_sum_generator, 0); }

      auto end() { return iterator(one_sum_generator, samples); }

    protected:
      const std::size_t samples;
      const std::function<MonteCarloSum()> one_sum_generator;
    };

  }

  template<typename TransformationType>
  struct MonteCarloTransform
  {
    using InputCoefficients = typename TransformationType::InputCoefficients;
    using OutputCoefficients = typename TransformationType::OutputCoefficients;

    explicit MonteCarloTransform(const TransformationType& transformation, const std::size_t samples = 100000)
      : size(samples), transformation(transformation) {}

  protected:
    const std::size_t size;
    const TransformationType transformation;

  public:
    template<typename InputDist, typename ... NoiseDist>
    auto operator()(const InputDist& in, const NoiseDist& ...n) const
    {
      using MSet = internal::MonteCarloSet<TransformationType, InputDist, NoiseDist...>;
      auto m_set = MSet(transformation, size, in, n...);
      auto binary_op = typename MSet::MonteCarloBinaryOp();
      using MSum = typename MSet::MonteCarloSum;

      MSum m_sum = std::reduce(std::execution::par_unseq, m_set.begin(), m_set.end(), MSet::zero(), binary_op);
      auto mean_output = strict(from_Euclidean(m_sum.y_E));
      auto out_covariance = strict(m_sum.yy / (size - 1.));
      auto cross_covariance = strict(m_sum.xy / (size - 1.));
      auto out = GaussianDistribution {mean_output, out_covariance};
      return std::tuple {std::move(out), std::move(cross_covariance)};
    }

  };


  template<typename TransformationType>
  auto make_MonteCarloTransform(TransformationType&& f, const std::size_t sample_size = 100000)
  {
    return MonteCarloTransform<std::decay_t<TransformationType>>(std::forward<TransformationType>(f), sample_size);
  };

}


#endif //OPENKALMAN_MONTECARLOTRANSFORM_H
