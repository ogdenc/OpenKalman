/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_LINEARIZEDTRANSFORM_H
#define OPENKALMAN_LINEARIZEDTRANSFORM_H

#include "transforms/support/LinearTransformBase.h"
#include "transforms/transformations/LinearTransformation.h"

namespace OpenKalman
{
  /**
   * @brief A linearized transform, using a 1st or 2nd order Taylor approximation of a linear transformation.
   */
  template<
    /// The transformation on which the transform is based.
    typename Transformation,
    /// Order of the Taylor approximation (1 or 2).
    int order = 1>
  struct LinearizedTransform;


  namespace internal
  {
    template<typename LinearizedTransformation, int order>
    struct LinearizedTransformFunction
    {
      using OutputCoeffs = typename LinearizedTransformation::OutputCoefficients;

      LinearizedTransformation transformation;

      LinearizedTransformFunction(const LinearizedTransformation& trans) : transformation(trans) {}

      LinearizedTransformFunction(LinearizedTransformation&& trans) noexcept : transformation(std::move(trans)) {}

    protected:
      /**
       * Add second-order moment terms, based on Hessian matrices.
       * @tparam Hessian An array of Hessian matrices. Must be accessible by bracket index, as in hessian[i].
       * Each matrix is a regular matrix type.
       * @tparam Dist Input distribution.
       * @return
       */
      template<typename Hessian, typename Dist>
      static auto second_order_term(const Hessian& hessian, const Dist& x)
      {
        constexpr auto output_dim = OutputCoeffs::size;
        //
        // Convert input distribution type to output distribution types, and initialize mean and covariance:
        using CovIn = typename MatrixTraits<typename DistributionTraits<Dist>::Covariance>::BaseMatrix;
        using CovOut = typename MatrixTraits<CovIn>::template SelfAdjointBaseType<triangle_type_of_v<CovIn>, output_dim>;
        using MeanOut = typename MatrixTraits<CovIn>::template StrictMatrix<output_dim, 1>;
        auto mean_terms = make_Mean<OutputCoeffs, MeanOut>();
        auto covariance_terms = make_Covariance<OutputCoeffs, CovOut>();
        //
        const auto P = covariance(x);
        for (std::size_t i = 0; i < output_dim; i++)
        {
          const auto P_hessian_i = P * hessian[i];
          mean_terms(i) = 0.5 * trace(P_hessian_i);
          for (int j = 0; j <= i; j++) // only need to fill out lower triangle of covariance:
          {
            covariance_terms(i, j) = 0.5 * trace(P_hessian_i * (P * hessian[j]));
          }
        }
        return GaussianDistribution<OutputCoeffs>(mean_terms, covariance_terms);
      }

      template<typename T1, typename T2, std::size_t...I>
      static auto zip_tuples_impl(const T1& t1, const T2& t2, std::index_sequence<I...>)
      {
        return strict((second_order_term(std::get<I>(t1), std::get<I>(t2)) + ...));
      }

      template<typename T1, typename T2>
      static constexpr auto zip_tuples(const T1& t1, const T2& t2)
      {
        return zip_tuples_impl(t1, t2, std::make_index_sequence<std::tuple_size_v<T1>>());
      }

    public:
      template<typename InputMean, typename ... NoiseMean>
      auto operator()(const InputMean& x, const NoiseMean& ... n) const
      {
        return std::tuple {transformation(x, n...), transformation.jacobian(x, n...)};
      }

      static constexpr bool correction = order > 1;

      template<typename Dist, typename ... Noise>
      auto add_correction(const Dist& x, const Noise& ... n) const
      {
        const auto hessians = transformation.hessian(mean(x), mean(n)...);
        static_assert(std::tuple_size_v<decltype(hessians)> == sizeof...(Noise) + 1, "Function must return one Hessian matrix for each input.");
        return zip_tuples(hessians, std::tuple {x, n...});
      }
    };

  }


  template<typename Transformation, int order>
  struct LinearizedTransform
    : internal::LinearTransformBase<typename Transformation::InputCoefficients, typename Transformation::OutputCoefficients,
      internal::LinearizedTransformFunction<Transformation, order>>
  {
    using InputCoefficients = typename Transformation::InputCoefficients;
    using OutputCoefficients = typename Transformation::OutputCoefficients;
    using Function = internal::LinearizedTransformFunction<Transformation, order>;
    using Base = internal::LinearTransformBase<InputCoefficients, OutputCoefficients, Function>;

    explicit LinearizedTransform(const Transformation& transformation)
      : Base(Function(transformation)) {}

    explicit LinearizedTransform(Transformation&& transformation)
      : Base(Function(std::move(transformation))) {}

  };

}

#endif //OPENKALMAN_LINEARIZEDTRANSFORM_H
