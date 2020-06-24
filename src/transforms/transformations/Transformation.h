/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TRANSFORMATION_H
#define OPENKALMAN_TRANSFORMATION_H

#include <type_traits>

namespace OpenKalman
{
  /**
   * @brief A transformation from one single-column vector to another, optionally incorporating noise terms.
   *
   * Models a transformation (linear or nonlinear) between two typed, single-column vectors.
   * The transformation can incorporate noise terms. These terms can either be constant in the form
   * of a typed matrix or mean, or stochastic in the form of a distribution.
   * @tparam InputCoefficients Coefficients of the input.
   * @tparam OutputCoefficients Coefficients of the output.
   * @tparam Function The transformation function, in the following form:
   * (Mean<InputCoefficients,...>, Mean<OutputCoefficients,...>, ...) -> Mean<OutputCoefficients,...>.
   * The first term is the input, the next term(s) represent nonlinear noise, and the final term is the output.
   */
  template<
    typename InputCoefficients,
    typename OutputCoefficients,
    typename Function>
  struct Transformation;


  ////////////////////
  //  Noise Traits  //
  ////////////////////

  template<typename T>
  struct is_noise : std::integral_constant<bool, is_Gaussian_distribution_v<T> or
    (is_typed_matrix_v<T> and is_column_vector_v<T> and not is_Euclidean_transformed_v<T>)> {};

  /// Helper template for is_noise.
  template<typename T>
  inline constexpr bool is_noise_v = is_noise<T>::value;


  namespace internal
  {
    template<typename Noise, typename T = void, typename Enable = void>
    struct NoiseTraits;

    template<typename Noise, typename T>
    struct NoiseTraits<Noise, T, std::enable_if_t<is_Gaussian_distribution_v<Noise>>>
      : MatrixTraits<typename DistributionTraits<Noise>::Mean> {};

    template<typename Noise, typename T>
    struct NoiseTraits<Noise, T, std::enable_if_t<is_typed_matrix_v<Noise>>>
      : MatrixTraits<Noise> {};

    template<typename Arg, std::enable_if_t<is_noise_v<Arg>, int> = 0>
    inline auto
    get_noise(Arg&& arg) noexcept
    {
      if constexpr(is_Gaussian_distribution_v<Arg>)
        return std::forward<Arg>(arg)();
      else
        return std::forward<Arg>(arg);
    }
  }


  //////////////////////
  //  Transformation  //
  //////////////////////

  template<
    typename InputCoefficients_,
    typename OutputCoefficients_,
    typename Func>
  struct Transformation
  {
    using InputCoefficients = InputCoefficients_; ///< Coefficients of the input.
    using OutputCoefficients = OutputCoefficients_; ///< Coefficients of the output.
    using Function = Func; ///< Transformation function type.

    const Function function; ///< The transformation function.

    /// Default constructor. Default-initializes the transformation function.
    Transformation() : function() {}

    /// Constructs transformation using a reference to a transformation function.
    Transformation(const Function& f) : function(f) {}

    template<
      typename M, typename ... Noise,
      std::enable_if_t<is_typed_matrix_v<M> and std::conjunction_v<is_noise<Noise>...>, int> = 0>
    auto operator()(M&& in, Noise&& ... noise) const
    {
      static_assert(is_column_vector_v<M>);
      static_assert(MatrixTraits<M>::columns == 1);
      static_assert(((internal::NoiseTraits<Noise>::columns == 1) and ...));
      static_assert(is_equivalent_v<typename MatrixTraits<M>::RowCoefficients, InputCoefficients>);
      static_assert(std::conjunction_v<is_equivalent<typename internal::NoiseTraits<Noise>::RowCoefficients, OutputCoefficients>...>);
      return function(std::forward<M>(in), internal::get_noise(std::forward<Noise>(noise))...);
    }
  };


  ////////////////////////////////////
  //  TransformationFunctionTraits  //
  ////////////////////////////////////

  /**
   * Traits of a transformation function from a Mean (and optional noise terms) to another Mean.
   * This will automatically derive input and output coefficients, if the function is not polymorphic.
   * @tparam Function The transformation function, which should transform one Mean to another.
   */
  template<typename Function, typename T = void, typename Enable = void>
  struct TransformationFunctionTraits {};

  namespace detail
  {
    template<typename Function, typename T = void, typename Enable1 = void, typename Enable2 = void>
    struct TransformationFunctionTraitsImpl {};

    template<typename In, typename Out, typename T, typename...Noise>
    struct TransformationFunctionTraitsImpl<std::function<Out(In, Noise...)>, T>
    {
      using type = T;
      using InputCoefficients = typename MatrixTraits<In>::RowCoefficients;
      using OutputCoefficients = typename MatrixTraits<Out>::RowCoefficients;
      static_assert(std::conjunction_v<is_equivalent<typename internal::NoiseTraits<Noise>::RowCoefficients, OutputCoefficients>...>);
      static_assert(is_column_vector_v<In>);
      static_assert(is_column_vector_v<Out>);
      static_assert(MatrixTraits<In>::columns == 1);
      static_assert(MatrixTraits<Out>::columns == 1);
      static_assert(((internal::NoiseTraits<Noise>::columns == 1) and ...));
    };
  }

  template<typename F, typename T>
  struct TransformationFunctionTraits<F, T, std::void_t<decltype(std::function(std::declval<F>()))>>
    : detail::TransformationFunctionTraitsImpl<decltype(std::function(std::declval<F>())), T> {};

  template<typename F, typename T>
  struct TransformationFunctionTraits<F&, T> : TransformationFunctionTraits<F, T> {};

  template<typename F, typename T>
  struct TransformationFunctionTraits<F&&, T> : TransformationFunctionTraits<F, T> {};

  template<typename F, typename T>
  struct TransformationFunctionTraits<const F, T> : TransformationFunctionTraits<F, T> {};


  ///////////////////////
  //  Deduction guide  //
  ///////////////////////

  /// Derive transformation template parameters from the function, if the function is not polymorphic.
  template<typename Function, typename TransformationFunctionTraits<Function, int>::type = 0>
  Transformation(Function&&)
  -> Transformation<
    typename TransformationFunctionTraits<Function>::InputCoefficients,
    typename TransformationFunctionTraits<Function>::OutputCoefficients,
    Function>;


  //////////////////////
  //  Make functions  //
  //////////////////////

  /// Make a Transformation from a transformation function.
  template<
    typename InputCoefficients, ///< Coefficients of the input.
    typename OutputCoefficients, ///< Coefficients of the output.
    typename Function> ///< Transformation function.
  auto make_Transformation(Function&& f)
  {
    return Transformation<InputCoefficients, OutputCoefficients, Function>(std::forward<Function>(f));
  };

  /// Make a transformation from a transformation function, deriving the coefficients.
  /// Substitution failure if the transformation function is not polymorphic.
  template<typename Function, typename TransformationFunctionTraits<Function, int>::type = 0>
  auto make_Transformation(Function&& f)
  {
    using InputCoefficients = typename TransformationFunctionTraits<Function>::InputCoefficients;
    using OutputCoefficients = typename TransformationFunctionTraits<Function>::OutputCoefficients;
    return Transformation<InputCoefficients, OutputCoefficients, Function>(std::forward<Function>(f));
  };

}


#endif //OPENKALMAN_TRANSFORMATION_H
