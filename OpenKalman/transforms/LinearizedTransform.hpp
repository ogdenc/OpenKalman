/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_LINEARIZEDTRANSFORM_HPP
#define OPENKALMAN_LINEARIZEDTRANSFORM_HPP


namespace OpenKalman
{
  /**
   * @brief A linearized transform, using a 1st or 2nd order Taylor approximation of a linear transformation.
   */
  template<unsigned int order = 1> ///< Order of the Taylor approximation (1 or 2).
  struct LinearizedTransform : internal::LinearTransformBase<LinearizedTransform<order>>
  {
  protected:
    using Base = internal::LinearTransformBase<LinearizedTransform>;
    friend Base;

    //-----------------------------------------------
    template<typename Transformation>
    struct TransformFunction
    {
    private:
      template<typename MeanType, typename Hessian, typename Cov, std::size_t...ints>
      static auto make_mean(const Hessian& hessian, const Cov& P, std::index_sequence<ints...>)
      {
        return MeanType {0.5 * trace(P * hessian[ints])...};
      }

      template<std::size_t i, std::size_t...js, typename Hessian, typename Cov>
      static constexpr auto make_cov_row(const Hessian& hessian, const Cov& P)
      {
        return std::tuple {0.5 * trace(P * hessian[i] * P * hessian[js])...};
      }

      template<typename CovType, typename Hessian, typename Cov, std::size_t...is, std::size_t...js>
      static auto make_cov(const Hessian& hessian, const Cov& P, std::index_sequence<is...>, std::index_sequence<js...>)
      {
        auto mat = std::tuple_cat(make_cov_row<is, js...>(hessian, P)...);
        return std::make_from_tuple<CovType>(std::move(mat));
      }

    protected:
      const Transformation& transformation;

      /**
       * Add second-order moment terms, based on Hessian matrices.
       * @tparam Hessian An array of Hessian matrices. Must be accessible by bracket index, as in hessian[i].
       * Each matrix is a regular matrix type.
       * @tparam Dist Input or noise distribution.
       * @return
       */
      template<typename OutputCoeffs, typename Hessian, typename Dist>
      static auto second_order_term(const Hessian& hessian, const Dist& x)
      {
        constexpr auto output_dim = std::tuple_size_v<Hessian>;
        static_assert(OutputCoeffs::size == output_dim);
        //
        // Convert input distribution type to output distribution types, and initialize mean and covariance:
        using CovIn = typename MatrixTraits<typename DistributionTraits<Dist>::Covariance>::BaseMatrix;
        using MeanOut = strict_matrix_t<CovIn, output_dim, 1>;
        using CovOut = typename MatrixTraits<CovIn>::template SelfAdjointBaseType<triangle_type_of_v<CovIn>, output_dim>;

        const auto P = covariance_of(x);
        const std::make_index_sequence<output_dim> ints;
        auto mean_terms = make_mean<Mean<OutputCoeffs, MeanOut>>(hessian, P, ints);
        auto cov_terms = make_cov<Covariance<OutputCoeffs, CovOut>>(hessian, P, ints, ints);
        return GaussianDistribution(mean_terms, cov_terms);
      }

      template<typename OutputCoeffs, typename T1, typename T2, std::size_t...I>
      static auto zip_tuples_impl(const T1& t1, const T2& t2, std::index_sequence<I...>)
      {
        return strict((second_order_term<OutputCoeffs>(std::get<I>(t1), std::get<I>(t2)) + ...));
      }

      template<typename OutputCoeffs, typename T1, typename T2>
      static constexpr auto zip_tuples(const T1& t1, const T2& t2)
      {
        static_assert(std::tuple_size_v<T1> == std::tuple_size_v<T2>);
        return zip_tuples_impl<OutputCoeffs>(t1, t2, std::make_index_sequence<std::tuple_size_v<T1>>());
      }

    public:
      TransformFunction(const Transformation& t) : transformation(t) {}

      template<typename InputMean, typename ... NoiseMean>
      auto operator()(const InputMean& x, const NoiseMean& ... n) const
      {
        return std::tuple {transformation(x, n...), transformation.jacobian(x, n...)};
      }

      static constexpr bool correction = order > 1;

      template<typename InputDist, typename ... Noise>
      auto add_correction(const InputDist& x, const Noise& ... n) const
      {
        using In_Mean = typename DistributionTraits<InputDist>::Mean;
        using Out_Mean = std::invoke_result_t<Transformation, In_Mean>;
        using OutputCoeffs = typename MatrixTraits<Out_Mean>::RowCoefficients;
        const auto hessians = transformation.hessian(mean_of(x), mean_of(n)...);
        return zip_tuples<OutputCoeffs>(hessians, std::tuple {x, n...});
      }
    };
    //-----------------------------------------------

  };


}

#endif //OPENKALMAN_LINEARIZEDTRANSFORM_HPP
