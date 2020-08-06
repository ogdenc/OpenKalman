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
   * @brief A transformation from one single-column vector to another.
   *
   * Models a transformation (linear or nonlinear) from one single-column vector to another.
   * The transformation takes an input vector, and optionally one or more perturbation terms. These can be
   * associated with noise, or translation, etc. The perturbation terms can either be constant single-column
   * vectors, or statistical distributions (in which case, the perturbation will be stochastic).
   * @tparam InputCoefficients Coefficients of the input.
   * @tparam OutputCoefficients Coefficients of the output.
   * @tparam Function The transformation function, in the following exemplary form:
   *   (Mean<InputCoefficients,...>, Mean<OutputCoefficients,...>, ...) -> Mean<OutputCoefficients,...>.
   *   The first term is the input, the next term(s) represent perturbation(s), and the final term is the output.
   * @tparam TaylorDerivatives Optional Taylor-series derivative functions, including the Jacobian and Hessian
   *   for the input and each perturbation.
   */
  template<
    typename InputCoefficients,
    typename OutputCoefficients,
    typename Function,
    typename...TaylorDerivatives>
  struct Transformation;


  //////////////
  //  Traits  //
  //////////////

  /// Whether an object is a linearized function (with defined Jacobian and optionally Hessian functions).
  template<typename T, std::size_t order = 1, typename Enable = void>
  struct is_linearized_function : std::false_type {};

  /// Helper template for is_linearized_function.
  template<typename T, std::size_t order>
  inline constexpr bool is_linearized_function_v = is_linearized_function<T, order>::value;

  template<typename T>
  struct is_linearized_function<T, 0,
    std::enable_if_t<
      std::is_member_function_pointer_v<decltype(&std::decay_t<T>::operator())> or
      std::is_function_v<T>>>
    : std::true_type {};

  template<typename T>
  struct is_linearized_function<T, 1,
    std::enable_if_t<std::is_member_function_pointer_v<decltype(&std::decay_t<T>::jacobian)> and
      is_linearized_function_v<T, 0>>>
    : std::true_type
    {
      static constexpr auto get_lambda(const T& t)
      {
        return [&t] (auto&&...inputs) { return t.jacobian(std::forward<decltype(inputs)>(inputs)...); };
      }
    };

  template<typename T>
  struct is_linearized_function<T, 2,
    std::enable_if_t<std::is_member_function_pointer_v<decltype(&std::decay_t<T>::hessian)> and
      is_linearized_function_v<T, 1>>>
    : std::true_type
    {
      static constexpr auto get_lambda(const T& t)
      {
        return [&t] (auto&&...inputs) { return t.hessian(std::forward<decltype(inputs)>(inputs)...); };
      }
    };

  template<typename In, typename Out, typename Tm, typename...Pm, std::size_t order>
  struct is_linearized_function<Transformation<In, Out, Tm, Pm...>, order,
    std::enable_if_t<order <= 2>> : std::true_type {};



  template<typename T>
  struct is_perturbation : std::integral_constant<bool, is_Gaussian_distribution_v<T> or
    (is_typed_matrix_v<T> and is_column_vector_v<T> and not is_Euclidean_transformed_v<T>)> {};

  /// Helper template for is_perturbation.
  template<typename T>
  inline constexpr bool is_perturbation_v = is_perturbation<T>::value;


  namespace internal
  {
    template<typename Noise, typename T = void, typename Enable = void>
    struct PerturbationTraits;

    template<typename Noise, typename T>
    struct PerturbationTraits<Noise, T, std::enable_if_t<is_Gaussian_distribution_v<Noise>>>
      : MatrixTraits<typename DistributionTraits<Noise>::Mean> {};

    template<typename Noise, typename T>
    struct PerturbationTraits<Noise, T, std::enable_if_t<is_typed_matrix_v<Noise>>>
      : MatrixTraits<Noise> {};

    template<typename Arg, std::enable_if_t<is_perturbation_v<Arg>, int> = 0>
    inline auto
    get_perturbation(Arg&& arg) noexcept
    {
      if constexpr(is_Gaussian_distribution_v<Arg>)
        return std::forward<Arg>(arg)();
      else
        return std::forward<Arg>(arg);
    }

    /// Create a tuple that replicates a value.
    template<std::size_t N, typename T>
    constexpr auto tuple_replicate(const T& t)
    {
      if constexpr(N < 1)
      {
        return std::tuple {};
      }
      else
      {
        return std::tuple_cat(std::make_tuple(t), tuple_replicate<N - 1>(t));
      }
    }

  }


  /////////////////////////////////////
  //  Non-linearized Transformation  //
  /////////////////////////////////////

  template<typename InputCoefficients_, typename OutputCoefficients_, typename Func>
  struct Transformation<InputCoefficients_, OutputCoefficients_, Func>
  {
    using InputCoefficients = InputCoefficients_; ///< Coefficients of the input.
    using OutputCoefficients = OutputCoefficients_; ///< Coefficients of the output.
    using Function = Func; ///< Transformation function type.

    /// Constructs transformation using a reference to a transformation function.
    Transformation(const Function& f = Function()) : function(f) {}

    /// Applies the transformation.
    template<typename M, typename ... Perturbations>
    auto operator()(M&& in, Perturbations&& ... perturbations) const
    {
      static_assert(is_column_vector_v<M>);
      static_assert((is_perturbation_v<Perturbations> and ...));
      static_assert(MatrixTraits<M>::columns == 1);
      static_assert(((internal::PerturbationTraits<Perturbations>::columns == 1) and ...));
      static_assert(is_equivalent_v<typename MatrixTraits<M>::RowCoefficients, InputCoefficients>);
      static_assert(std::conjunction_v<
        is_equivalent<typename internal::PerturbationTraits<Perturbations>::RowCoefficients, OutputCoefficients>...>);
      return function(std::forward<M>(in), internal::get_perturbation(std::forward<Perturbations>(perturbations))...);
    }

  protected:
    const Function function; ///< The transformation function.
  };


  /////////////////////////////////////////////
  //  First-Order Linearized Transformation  //
  /////////////////////////////////////////////

  template<typename InputCoefficients, typename OutputCoefficients, typename Function, typename JacobianFunc>
  struct Transformation<InputCoefficients, OutputCoefficients, Function, JacobianFunc>
    : Transformation<InputCoefficients, OutputCoefficients, Function>
  {
    using JacobianFunction = JacobianFunc;
    using Base = Transformation<InputCoefficients, OutputCoefficients, Function>;

  protected:
    static auto default_Jacobian(const Function& f)
    {
      if constexpr (is_linearized_function_v<Function, 1>) return is_linearized_function<Function, 1>::get_lambda(f);
      else return JacobianFunc();
    }

  public:
    /// Default constructor.
    Transformation()
      : Base(Function()), jacobian_fun(JacobianFunc()) {}

    /// Constructs transformation using a transformation function.
    Transformation(const Function& f)
      : Base(f), jacobian_fun(default_Jacobian(f)) {}

    /// Constructs transformation using a transformation function and a Jacobian function.
    Transformation(const Function& f, const JacobianFunction& j)
      : Base(f), jacobian_fun(j) {}

    /// Returns a tuple of the Jacobians for the input and each perturbation term.
    template<typename In, typename ... Perturbations>
    auto jacobian(In&& in, Perturbations&&...ps) const
    {
      static_assert(is_column_vector_v<In>);
      static_assert((is_perturbation_v<Perturbations> and ...));
      static_assert(is_equivalent_v<typename MatrixTraits<In>::RowCoefficients, InputCoefficients>);
      static_assert(std::conjunction_v<
        is_equivalent<typename internal::PerturbationTraits<Perturbations>::RowCoefficients, OutputCoefficients>...>);
      return jacobian_fun(std::forward<In>(in), internal::get_perturbation(std::forward<Perturbations>(ps))...);
    }

    /// Returns a tuple of Hessian matrices for the input and each perturbation term. In this case, they are zero matrices.
    template<typename In, typename ... Perturbations>
    auto hessian(In&&, Perturbations&&...) const
    {
      static_assert(is_column_vector_v<In>);
      static_assert((is_perturbation_v<Perturbations> and ...));
      static_assert(is_equivalent_v<typename MatrixTraits<In>::RowCoefficients, InputCoefficients>);
      static_assert(std::conjunction_v<
        is_equivalent<typename internal::PerturbationTraits<Perturbations>::RowCoefficients, OutputCoefficients>...>);
      constexpr std::size_t input_size = InputCoefficients::size;
      constexpr std::size_t output_size = OutputCoefficients::size;

      using HessianMatrixInBase = typename MatrixTraits<In>::template StrictMatrix<input_size, input_size>;
      using HessianMatrixIn = TypedMatrix<InputCoefficients, InputCoefficients, HessianMatrixInBase>;
      using HessianArrayIn = std::array<HessianMatrixIn, output_size>;
      HessianArrayIn a;
      a.fill(HessianMatrixIn::zero());
      if constexpr (sizeof...(Perturbations) >= 1)
      {
        using HessianMatrixNoiseBase = typename MatrixTraits<In>::template StrictMatrix<output_size, output_size>;
        using HessianMatrixNoise = TypedMatrix<OutputCoefficients, OutputCoefficients, HessianMatrixNoiseBase>;
        using HessianArrayNoise = std::array<HessianMatrixNoise, output_size>;
        HessianArrayNoise an;
        an.fill(HessianMatrixNoise::zero());
        return std::tuple_cat(std::tuple(std::move(a)), internal::tuple_replicate<sizeof...(Perturbations)>(std::move(an)));
      }
      else
      {
        return std::tuple(std::move(a));
      }
    }

  protected:
    const JacobianFunction jacobian_fun;
  };


  template<
    typename InputCoefficients,
    typename OutputCoefficients,
    typename Function,
    typename JacobianFunction,
    typename HessianFunc>
  struct Transformation<InputCoefficients, OutputCoefficients, Function, JacobianFunction, HessianFunc>
    : Transformation<InputCoefficients, OutputCoefficients, Function, JacobianFunction>
  {
    using HessianFunction = HessianFunc;
    using Base = Transformation<InputCoefficients, OutputCoefficients, Function, JacobianFunction>;

  protected:
    static auto default_Hessian(const Function& f)
    {
      if constexpr (is_linearized_function_v<Function, 2>) return is_linearized_function<Function, 2>::get_lambda(f);
      else return HessianFunc();
    }

  public:
    /// Constructs transformation using a transformation function and a Jacobian and Hessian functions.
    /// Default constructor.
    Transformation()
      : Base(), hessian_fun(HessianFunc()) {}

    /// Constructs transformation using a transformation function.
    Transformation(const Function& f)
      : Base(f), hessian_fun(default_Hessian(f)) {}

    /// Constructs transformation using a transformation function and a Jacobian function.
    Transformation(const Function& f, const JacobianFunction& j)
      : Base(f, j), hessian_fun(default_Hessian(f)) {}

    /// Constructs transformation using a transformation function and a Jacobian function.
    Transformation(const Function& f, const JacobianFunction& j, const HessianFunction& h)
      : Base(f, j), hessian_fun(h) {}

    /// Returns a tuple of Hessian matrices for the input and each perturbation term.
    template<typename In, typename ... Perturbations>
    auto hessian(In&& in, Perturbations&&...ps) const
    {
      static_assert(is_column_vector_v<In>);
      static_assert((is_perturbation_v<Perturbations> and ...));
      static_assert(is_equivalent_v<typename MatrixTraits<In>::RowCoefficients, InputCoefficients>);
      static_assert(std::conjunction_v<
        is_equivalent<typename internal::PerturbationTraits<Perturbations>::RowCoefficients, OutputCoefficients>...>);
      return hessian_fun(std::forward<In>(in), internal::get_perturbation(std::forward<Perturbations>(ps))...);
    }

  protected:
    const HessianFunction hessian_fun;
  };


  ////////////////////////////////////
  //  TransformationFunctionTraits  //
  ////////////////////////////////////

  /**
   * Traits of a transformation function.
   * This will automatically derive input and output coefficients, if the function is not polymorphic.
   * @tparam Function The transformation function, which should transform one Mean to another.
   */
  template<typename Function, typename T = void, typename Enable = void>
  struct TransformationFunctionTraits {};

  namespace detail
  {
    template<typename Function, typename T = void, typename Enable1 = void, typename Enable2 = void>
    struct TransformationFunctionTraitsImpl {};

    template<typename In, typename Out, typename T, typename...Perturbations>
    struct TransformationFunctionTraitsImpl<std::function<Out(In, Perturbations...)>, T>
    {
      using type = T;
      using InputCoefficients = typename MatrixTraits<In>::RowCoefficients;
      using OutputCoefficients = typename MatrixTraits<Out>::RowCoefficients;
      static_assert(std::conjunction_v<
        is_equivalent<typename internal::PerturbationTraits<Perturbations>::RowCoefficients, OutputCoefficients>...>);
      static_assert(is_column_vector_v<In>);
      static_assert(is_column_vector_v<Out>);
      static_assert(MatrixTraits<In>::columns == 1);
      static_assert(MatrixTraits<Out>::columns == 1);
      static_assert(((internal::PerturbationTraits<Perturbations>::columns == 1) and ...));
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
  //  Deduction guides  //
  ///////////////////////

  /// Derive transformation template parameters from the function. (Substitution failure if function is polymorphic.)
  template<typename Function,
    typename TransformationFunctionTraits<Function, int>::type = 0,
    std::enable_if_t<not is_linearized_function_v<Function, 1>, int> = 0>
  Transformation(const Function&)
  -> Transformation<
    typename TransformationFunctionTraits<Function>::InputCoefficients,
    typename TransformationFunctionTraits<Function>::OutputCoefficients,
    Function>;

  /// Derive transformation template parameters from the function. (Substitution failure if function is polymorphic.)
  template<typename Function, typename JacobianFunction, typename...TaylorDerivatives,
    typename TransformationFunctionTraits<Function, int>::type = 0,
    std::enable_if_t<not is_linearized_function_v<Function, 1>, int> = 0>
  Transformation(const Function&, const JacobianFunction&, const TaylorDerivatives&...)
  -> Transformation<
    typename TransformationFunctionTraits<Function>::InputCoefficients,
    typename TransformationFunctionTraits<Function>::OutputCoefficients,
    Function,
    JacobianFunction,
    TaylorDerivatives...>;

  /// Derive transformation template parameters from a first-order linearized function.
  template<typename Function, typename TransformationFunctionTraits<Function, int>::type = 0,
    std::enable_if_t<is_linearized_function_v<Function, 1> and not is_linearized_function_v<Function, 2>, int> = 0>
  Transformation(const Function&)
  -> Transformation<
    typename TransformationFunctionTraits<Function>::InputCoefficients,
    typename TransformationFunctionTraits<Function>::OutputCoefficients,
    Function,
    decltype(is_linearized_function<Function, 1>::get_lambda(std::declval<Function>()))>;

  /// Derive transformation template parameters from a second-order linearized function.
  template<typename Function, typename TransformationFunctionTraits<Function, int>::type = 0,
    std::enable_if_t<is_linearized_function_v<Function, 2>, int> = 0>
  Transformation(const Function&)
  -> Transformation<
    typename TransformationFunctionTraits<Function>::InputCoefficients,
    typename TransformationFunctionTraits<Function>::OutputCoefficients,
    Function,
    decltype(is_linearized_function<Function, 1>::get_lambda(std::declval<Function>())),
    decltype(is_linearized_function<Function, 2>::get_lambda(std::declval<Function>()))>;


  //////////////////////
  //  Make functions  //
  //////////////////////

  /// Make a Transformation from a transformation function (and optionally one or more Taylor derivatives).
  template<
    typename InputCoefficients, ///< Coefficients of the input.
    typename OutputCoefficients, ///< Coefficients of the output.
    typename Function, ///< Transformation function.
    std::enable_if_t<not is_linearized_function_v<Function, 1>, int> = 0>
  auto make_Transformation(const Function& f)
  {
    return Transformation<InputCoefficients, OutputCoefficients, Function>(f);
  };

  /// Make a Transformation from a transformation function (and optionally one or more Taylor derivatives).
  template<
    typename InputCoefficients, ///< Coefficients of the input.
    typename OutputCoefficients, ///< Coefficients of the output.
    typename Function, ///< Transformation function.
    typename JacobianFunction,
    typename...TaylorDerivatives,
    std::enable_if_t<not is_linearized_function_v<Function, 1>, int> = 0>
  auto make_Transformation(const Function& f, const JacobianFunction& j, const TaylorDerivatives&...ds)
  {
    return Transformation<InputCoefficients, OutputCoefficients, Function, TaylorDerivatives...>(f, j, ds...);
  };

  /// Make a Transformation from a first-order linearized transformation function.
  template<
    typename InputCoefficients, ///< Coefficients of the input.
    typename OutputCoefficients, ///< Coefficients of the output.
    typename Function, ///< Transformation function.
    std::enable_if_t<is_linearized_function_v<Function, 1> and not is_linearized_function_v<Function, 2>, int> = 0>
  auto make_Transformation(const Function& f)
  {
    return Transformation<InputCoefficients, OutputCoefficients, Function,
      decltype(is_linearized_function<Function, 1>::get_lambda(std::declval<Function>()))>(f);
  };

  /// Make a Transformation from a second-order linearized transformation function.
  template<
    typename InputCoefficients, ///< Coefficients of the input.
    typename OutputCoefficients, ///< Coefficients of the output.
    typename Function, ///< Transformation function.
    std::enable_if_t<is_linearized_function_v<Function, 2>, int> = 0>
  auto make_Transformation(const Function& f)
  {
    return Transformation<InputCoefficients, OutputCoefficients, Function,
      decltype(is_linearized_function<Function, 1>::get_lambda(std::declval<Function>())),
      decltype(is_linearized_function<Function, 2>::get_lambda(std::declval<Function>()))>(f);
  };

  /// Make a transformation from a transformation function, deriving the coefficients.
  /// Substitution failure if the transformation function is polymorphic.
  template<
    typename Function,
    typename...TaylorDerivatives,
    typename TransformationFunctionTraits<Function, int>::type = 0,
    std::enable_if_t<not is_linearized_function_v<Function, 1>, int> = 0>
  auto make_Transformation(const Function& f, const TaylorDerivatives&...ds)
  {
    using InputCoefficients = typename TransformationFunctionTraits<Function>::InputCoefficients;
    using OutputCoefficients = typename TransformationFunctionTraits<Function>::OutputCoefficients;
    return Transformation<InputCoefficients, OutputCoefficients, Function, TaylorDerivatives...>(f, ds...);
  };

  /// Make a transformation from a first-order linearized transformation function, deriving the coefficients.
  /// Substitution failure if the transformation function is polymorphic.
  template<
    typename Function,
    typename TransformationFunctionTraits<Function, int>::type = 0,
    std::enable_if_t<is_linearized_function_v<Function, 1> and not is_linearized_function_v<Function, 2>, int> = 0>
  auto make_Transformation(const Function& f)
  {
    using InputCoefficients = typename TransformationFunctionTraits<Function>::InputCoefficients;
    using OutputCoefficients = typename TransformationFunctionTraits<Function>::OutputCoefficients;
    return Transformation<InputCoefficients, OutputCoefficients, Function,
      decltype(is_linearized_function<Function, 1>::get_lambda(std::declval<Function>()))>(f);
  };

  /// Make a transformation from a first-order linearized transformation function, deriving the coefficients.
  /// Substitution failure if the transformation function is polymorphic.
  template<
    typename Function,
    typename TransformationFunctionTraits<Function, int>::type = 0,
    std::enable_if_t<is_linearized_function_v<Function, 2>, int> = 0>
  auto make_Transformation(const Function& f)
  {
    using InputCoefficients = typename TransformationFunctionTraits<Function>::InputCoefficients;
    using OutputCoefficients = typename TransformationFunctionTraits<Function>::OutputCoefficients;
    return Transformation<InputCoefficients, OutputCoefficients, Function,
      decltype(is_linearized_function<Function, 1>::get_lambda(std::declval<Function>())),
      decltype(is_linearized_function<Function, 2>::get_lambda(std::declval<Function>()))>(f);
  };

}


#endif //OPENKALMAN_TRANSFORMATION_H
