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
 * \brief Definition of LinearizedTransform.
 */

#ifndef OPENKALMAN_LINEARIZEDTRANSFORM_HPP
#define OPENKALMAN_LINEARIZEDTRANSFORM_HPP


namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  /**
   * \brief A linearized transform, using a 1st or 2nd order Taylor approximation of a linear tests.
   * \tparam order The maximum order of the Taylor approximation (1 or 2). If a tests function does not
   * define a Hessian matrix, the order will be treated as 1, even if it is defined here as 2.
   */
  template<unsigned int order = 1>
  class LinearizedTransform;


  namespace internal
  {
    template<unsigned int order, typename F>
    struct needs_additive_correction<LinearizedTransform<order>, F> : std::bool_constant<
      (order >= 2) and linearized_function<F, 2>> {};
  }


  template<unsigned int order>
  class LinearizedTransform : public oin::LinearTransformBase<LinearizedTransform<order>>
  {
    using Base = oin::LinearTransformBase<LinearizedTransform>;
    friend Base;


    /**
     * \internal
     * \brief The underlying transform model for LinearizedTransform.
     * \tparam Trans The tests function.
     */
    template<typename Trans>
    struct TransformModel
    {

    private:

      template<typename MeanType, typename Hessian, typename Cov, std::size_t...ints>
      static auto construct_mean(const Hessian& hessian, const Cov& P, std::index_sequence<ints...>)
      {
        return MeanType {0.5 * trace(P * hessian[ints])...};
      }


      template<std::size_t i, std::size_t...js, typename Hessian, typename Cov>
      static constexpr auto construct_cov_row(const Hessian& hessian, const Cov& P)
      {
        return std::tuple {0.5 * trace(P * hessian[i] * P * hessian[js])...};
      }


      template<typename CovType, typename Hessian, typename Cov, std::size_t...is, std::size_t...js>
      static auto construct_cov(const Hessian& hessian, const Cov& P, std::index_sequence<is...>, std::index_sequence<js...>)
      {
        auto mat = std::tuple_cat(construct_cov_row<is, js...>(hessian, P)...);
        return std::make_from_tuple<CovType>(std::move(mat));
      }


      /*
       * \brief Add second-order moment terms, based on Hessian matrices (a tensor of order (1,2)).
       * \tparam Hessian An array of Hessian matrices. Must be accessible by bracket index, as in hessian[i].
       * Each matrix is a regular matrix type.
       * \tparam Dist Input or noise distribution.
       * \return
       */
      template<typename OutputCoeffs, typename Hessian, typename Dist>
      static auto second_order_term(const Hessian& hessian, const Dist& x)
      {
        constexpr auto output_dim = std::tuple_size_v<Hessian>;
        static_assert(output_dim == OutputCoeffs::dimension);
        static_assert(order >= 2);

        // Convert input distribution type to output distribution types, and initialize mean and covariance:
        using CovIn = nested_matrix_of_t<typename DistributionTraits<Dist>::Covariance>;
        using MeanOut = equivalent_dense_writable_matrix_t<CovIn, output_dim, 1>;
        constexpr TriangleType tri = triangle_type_of_v<typename MatrixTraits<CovIn>::template TriangularMatrixFrom<>>;
        using CovOut = typename MatrixTraits<CovIn>::template SelfAdjointMatrixFrom<tri, output_dim>;

        auto P = covariance_of(x);
        const std::make_index_sequence<output_dim> ints;
        auto mean_terms = construct_mean<Mean<OutputCoeffs, MeanOut>>(hessian, P, ints);
        auto cov_terms = construct_cov<Covariance<OutputCoeffs, CovOut>>(hessian, P, ints, ints);
        return GaussianDistribution {std::move(mean_terms), std::move(cov_terms)};
      }


      template<typename OutputCoeffs, typename T1, typename T2, std::size_t...I>
      static auto zip_tuples_impl(T1&& t1, T2&& t2, std::index_sequence<I...>)
      {
        static_assert(order >= 2);
        return make_self_contained((second_order_term<OutputCoeffs>(
          std::get<I>(std::forward<T1>(t1)), std::get<I>(std::forward<T2>(t2))) + ...));
      }


      template<typename OutputCoeffs, typename T1, typename T2>
      static constexpr auto zip_tuples(T1&& t1, T2&& t2)
      {
        static_assert(order >= 2);
        constexpr auto s = std::tuple_size_v<std::decay_t<T1>>;
        static_assert(s == std::tuple_size_v<std::decay_t<T2>>);
        return zip_tuples_impl<OutputCoeffs>(std::forward<T1>(t1), std::forward<T2>(t2), std::make_index_sequence<s>());
      }

    public:

      /**
       * \brief Constructor
       * \param t A tests function.
       */
      TransformModel(const Trans& t) : transformation(t) {}


      /**
       * \tparam InputMean The input mean.
       * \tparam NoiseMean Zero or more noise means.
       * \return A tuple comprising the output mean and the Jacobians corresponding to the input mean and noise terms.
       */
#ifdef __cpp_concepts
      template<transformation_input InputMean, perturbation ... NoiseMean>
#else
      template<typename InputMean, typename ... NoiseMean, std::enable_if_t<transformation_input<InputMean> and
        (perturbation<NoiseMean> and ...), int> = 0>
#endif
      auto operator()(const InputMean& x, const NoiseMean& ... n) const
      {
        return std::tuple {transformation(x, n...), transformation.jacobian(x, n...)};
      }


      /**
       * \tparam InputDist The input distribution.
       * \tparam Noise Zero or more noise distributions.
       * \brief Add a correction term to the transform's output distribution.
       */
#ifdef __cpp_concepts
      template<gaussian_distribution InputDist, gaussian_distribution ... Noise> requires (order >= 2)
#else
      template<typename InputDist, typename ... Noise, std::enable_if_t<
        gaussian_distribution<InputDist> and (gaussian_distribution<Noise> and ...) and (order >= 2), int> = 0>
#endif
      auto add_correction(const InputDist& x, const Noise& ... n) const
      {
        using In_Mean = typename DistributionTraits<InputDist>::Mean;
        using Out_Mean = std::invoke_result_t<Trans, In_Mean>;
        using OutputCoeffs = typename MatrixTraits<Out_Mean>::RowCoefficients;
        auto hessians = transformation.hessian(mean_of(x), mean_of(n)...);

        return make_self_contained(zip_tuples<OutputCoeffs>(std::move(hessians), std::forward_as_tuple(x, n...)));
      }

    private:

      const Trans& transformation;

    };

  };


}

#endif //OPENKALMAN_LINEARIZEDTRANSFORM_HPP
