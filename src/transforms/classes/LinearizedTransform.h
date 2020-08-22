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


namespace OpenKalman
{
  /**
   * @brief A linearized transform, using a 1st or 2nd order Taylor approximation of a linear transformation.
   */
  template<
    unsigned int order = 1> ///< Order of the Taylor approximation (1 or 2).
  struct LinearizedTransform : internal::LinearTransformBase
  {
  protected:
    using Base = internal::LinearTransformBase;

    template<typename Transformation>
    struct TransformFunction
    {
    protected:
      using OutputCoeffs = typename Transformation::OutputCoefficients;

      const Transformation& transformation;

    private:
      template<typename OutputCoeffs, typename MeanOut, typename Hessian, typename Cov, std::size_t...ints>
      static auto make_mean(const Hessian& hessian, const Cov& P, std::index_sequence<ints...>)
      {
        return Mean<OutputCoeffs, MeanOut> {0.5 * trace(P * hessian[ints])...};
      }

      template<std::size_t i, std::size_t...js, typename Hessian, typename Cov>
      static constexpr auto make_cov_row(const Hessian& hessian, const Cov& P)
      {
        return std::tuple {0.5 * trace(P * hessian[i] * P * hessian[js])...};
      }

      template<typename OutputCoeffs, typename CovOut, typename Hessian, typename Cov, std::size_t...is, std::size_t...js>
      static auto make_cov(const Hessian& hessian, const Cov& P, std::index_sequence<is...>, std::index_sequence<js...>)
      {
        auto mat = std::tuple_cat(make_cov_row<is, js...>(hessian, P)...);
        return std::make_from_tuple<Covariance<OutputCoeffs, CovOut>>(std::move(mat));
      }

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
        using MeanOut = typename MatrixTraits<CovIn>::template StrictMatrix<output_dim, 1>;
        using CovOut = typename MatrixTraits<CovIn>::template SelfAdjointBaseType<triangle_type_of_v<CovIn>, output_dim>;
        const auto P = covariance(x);
        const std::make_index_sequence<output_dim> ints;
        auto mean_terms = make_mean<OutputCoeffs, MeanOut>(hessian, P, ints);
        auto cov_terms = make_cov<OutputCoeffs, CovOut>(hessian, P, ints, ints);
        return GaussianDistribution(mean_terms, cov_terms);
      }

      template<typename T1, typename T2, std::size_t...I>
      static auto zip_tuples_impl(const T1& t1, const T2& t2, std::index_sequence<I...>)
      {
        return strict((second_order_term(std::get<I>(t1), std::get<I>(t2)) + ...));
      }

      template<typename T1, typename T2>
      static constexpr auto zip_tuples(const T1& t1, const T2& t2)
      {
        static_assert(std::tuple_size_v<T1> == std::tuple_size_v<T2>);
        return zip_tuples_impl(t1, t2, std::make_index_sequence<std::tuple_size_v<T1>>());
      }

    public:
      TransformFunction(const Transformation& t) : transformation(t) {}

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
        return zip_tuples(hessians, std::tuple {x, n...});
      }
    };

  public:
    /**
     * Perform a linearized transform from one statistical distribution to another.
     * @tparam Transformation The transformation on which the transform is based.
     * @tparam InputDist Input distribution.
     * @tparam NoiseDist Noise distribution.
     **/
    template<typename Transformation, typename InputDist, typename ... NoiseDist,
      std::enable_if_t<std::conjunction_v<is_distribution<InputDist>, is_distribution<NoiseDist>...>, int> = 0>
    auto operator()(const Transformation& transformation, const InputDist& in, const NoiseDist& ...n) const
    {
      return Base::transform(TransformFunction {transformation}, in, n...);
    }

  };


}

#endif //OPENKALMAN_LINEARIZEDTRANSFORM_H
