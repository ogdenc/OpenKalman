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
#include "variables/support/TypedMatrixOverloads.h"
#include "distributions/DistributionTraits.h"

namespace OpenKalman
{
  /**
   * @brief A transformation from one Mean to another, optionally incorporating noise terms.
   *
   * Models a transformation (linear or nonlinear) between two Mean vectors.
   * The transformation can incorporate noise terms. These terms can either be constant in the form
   * of a mean vector, or stochastic in the form of a distribution.
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


  ///////////////////
  //  NoiseTraits  //
  ///////////////////

  template<typename Noise, typename T = void, typename Enable = void>
  struct NoiseTraits;

  template<typename Noise, typename T>
  struct NoiseTraits<Noise, T, std::enable_if_t<is_Gaussian_distribution_v<Noise>>>
    : MatrixTraits<typename DistributionTraits<Noise>::Mean>
  {
    using stochastic_type = T;
  };

  template<typename Noise, typename T>
  struct NoiseTraits<Noise, T, std::enable_if_t<is_typed_matrix_v<Noise>>>
    : MatrixTraits<Noise>
  {
    static_assert(is_column_vector_v<Noise>);
    static_assert(not is_Euclidean_transformed_v<Noise>);
    using constant_type = T;
  };

  template<typename Arg, typename NoiseTraits<Arg, int>::stochastic_type = 0>
  inline auto
  get_noise(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg)();
  }

  template<typename Arg, typename NoiseTraits<Arg, int>::constant_type = 0>
  constexpr decltype(auto)
  get_noise(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
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
      std::enable_if_t<is_typed_matrix_v<M>, int> = 0,
      typename = std::void_t<typename NoiseTraits<Noise>::type...>>
    auto operator()(M&& in, Noise&& ... noise) const
    {
      static_assert(is_column_vector_v<M>);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<M>::RowCoefficients, InputCoefficients>);
      static_assert(std::conjunction_v<OpenKalman::is_equivalent_v<typename NoiseTraits<Noise>::RowCoefficients, OutputCoefficients>...>);
      return function(std::forward<M>(in), get_noise(std::forward<Noise>(noise))...);
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
      static_assert(std::conjunction_v<std::is_same<typename NoiseTraits<Noise>::RowCoefficients, OutputCoefficients>...>);
      static_assert(is_column_vector_v<In>);
      static_assert(is_column_vector_v<Out>);
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
