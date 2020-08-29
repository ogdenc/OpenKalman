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
   * @tparam Function The transformation function, in the following exemplary form:
   *   (Mean<InputCoefficients,...>, Mean<OutputCoefficients,...>, ...) -> Mean<OutputCoefficients,...>.
   *   The first term is the input, the next term(s) represent perturbation(s), and the final term is the output.
   * @tparam TaylorDerivatives Optional Taylor-series derivative functions, including the Jacobian and Hessian
   *   for the input and each perturbation.
   */
  template<
    typename Function,
    typename...TaylorDerivatives>
  struct Transformation;


  //////////////
  //  Traits  //
  //////////////

  /// Whether an object is a linearized function (with defined Jacobian and optionally Hessian functions).
  template<typename T, std::size_t order = 1, typename Enable = void>
  struct is_linearized_function : std::false_type {};

  template<typename T, std::size_t order>
  struct is_linearized_function<T&, order> : is_linearized_function<T, order> {};

  template<typename T, std::size_t order>
  struct is_linearized_function<T&&, order> : is_linearized_function<T, order> {};

  template<typename T, std::size_t order>
  struct is_linearized_function<const T, order> : is_linearized_function<T, order> {};

  /// Helper template for is_linearized_function.
  template<typename T, std::size_t order>
  inline constexpr bool is_linearized_function_v = is_linearized_function<T, order>::value;

  template<typename T>
  struct is_linearized_function<T, 0,
    std::enable_if_t<
      not std::is_reference_v<T> and not std::is_const_v<T> and
      (std::is_member_function_pointer_v<decltype(&std::decay_t<T>::operator())> or
      std::is_function_v<T>)>>
    : std::true_type {};

  template<typename T>
  struct is_linearized_function<T, 1,
    std::enable_if_t<
      not std::is_reference_v<T> and not std::is_const_v<T> and
      (std::is_member_function_pointer_v<decltype(&T::jacobian)> and
      is_linearized_function_v<T, 0>)>>
    : std::true_type
  {
    static constexpr auto get_lambda(const T& t)
    {
      return [&t] (auto&&...inputs) { return t.jacobian(std::forward<decltype(inputs)>(inputs)...); };
    }
  };

  template<typename T>
  struct is_linearized_function<T, 2,
    std::enable_if_t<
      not std::is_reference_v<T> and not std::is_const_v<T> and
      (std::is_member_function_pointer_v<decltype(&T::hessian)> and
      is_linearized_function_v<T, 1>)>>
    : std::true_type
  {
    static constexpr auto get_lambda(const T& t)
    {
      return [&t] (auto&&...inputs) { return t.hessian(std::forward<decltype(inputs)>(inputs)...); };
    }
  };

  template<typename Function, typename...TaylorDerivatives>
  struct is_linearized_function<Transformation<Function, TaylorDerivatives...>, 0> : std::true_type {};

  template<typename Function, typename Jacobian, typename...TaylorDerivatives>
  struct is_linearized_function<Transformation<Function, Jacobian, TaylorDerivatives...>, 1> : std::true_type
  {
    static constexpr auto get_lambda(const Transformation<Function, Jacobian, TaylorDerivatives...>& t)
    {
      return [&t] (auto&&...inputs) { return t.jacobian(std::forward<decltype(inputs)>(inputs)...); };
    }
  };

  template<typename Function, typename Jacobian, typename...TaylorDerivatives>
  struct is_linearized_function<Transformation<Function, Jacobian, TaylorDerivatives...>, 2> : std::true_type
  {
    static constexpr auto get_lambda(const Transformation<Function, Jacobian, TaylorDerivatives...>& t)
    {
      return [&t] (auto&&...inputs) { return t.hessian(std::forward<decltype(inputs)>(inputs)...); };
    }
  };


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


  /// A tuple of zero-filled arrays of Hessian matrices, based on the input and each perturbation term.
  template<typename OutputCoefficients, typename In, typename ... Perturbations>
  inline auto zero_hessian()
  {
    static_assert(is_column_vector_v<In>);
    static_assert((is_perturbation_v<Perturbations> and ...));

    using InputCoefficients = typename MatrixTraits<In>::RowCoefficients;
    constexpr std::size_t input_size = InputCoefficients::size;
    constexpr std::size_t output_size = OutputCoefficients::size;

    using HessianMatrixInBase = strict_matrix_t<In, input_size, input_size>;
    using HessianMatrixIn = TypedMatrix<InputCoefficients, InputCoefficients, HessianMatrixInBase>;
    using HessianArrayIn = std::array<HessianMatrixIn, output_size>;
    HessianArrayIn a;
    a.fill(HessianMatrixIn::zero());
    if constexpr (sizeof...(Perturbations) >= 1)
    {
      using HessianMatrixNoiseBase = strict_matrix_t<In, output_size, output_size>;
      using HessianMatrixNoise = TypedMatrix<OutputCoefficients, OutputCoefficients, HessianMatrixNoiseBase>;
      using HessianArrayNoise = std::array<HessianMatrixNoise, output_size>;
      HessianArrayNoise an;
      an.fill(HessianMatrixNoise::zero());
      return std::tuple_cat(std::tuple {std::move(a)}, internal::tuple_replicate<sizeof...(Perturbations)>(std::move(an)));
    }
    else
    {
      return std::tuple {std::move(a)};
    }
  }


  /////////////////////////////////////
  //  Non-linearized Transformation  //
  /////////////////////////////////////

  template<typename Func>
  struct Transformation<Func>
  {
    using Function = Func; ///< Transformation function type.

    /// Constructs transformation using a reference to a transformation function.
    Transformation(const Function& f = Function()) : function(f) {}

    /// Applies the transformation.
    template<typename In, typename ... Perturbations>
    auto operator()(In&& in, Perturbations&& ... ps) const
    {
      static_assert(is_column_vector_v<In>);
      static_assert((is_perturbation_v<Perturbations> and ...));
      static_assert(MatrixTraits<In>::columns == 1);
      static_assert(((internal::PerturbationTraits<Perturbations>::columns == 1) and ...));

      return function(std::forward<In>(in), internal::get_perturbation(std::forward<Perturbations>(ps))...);
    }

  protected:
    const Function function; ///< The transformation function.
  };


  /////////////////////////////////////////////
  //  First-Order Linearized Transformation  //
  /////////////////////////////////////////////

  template<typename Function, typename JacobianFunc>
  struct Transformation<Function, JacobianFunc> : Transformation<Function>
  {
    using JacobianFunction = JacobianFunc;
    using Base = Transformation<Function>;

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

      return jacobian_fun(std::forward<In>(in), internal::get_perturbation(std::forward<Perturbations>(ps))...);
    }

    /// Returns a tuple of Hessian matrices for the input and each perturbation term. In this case, they are zero matrices.
    template<typename In, typename ... Perturbations>
    auto hessian(In&&, Perturbations&&...) const
    {
      using Out_Mean = std::invoke_result_t<Function, In&&, decltype(internal::get_perturbation(std::declval<Perturbations&&>()))...>;
      using OutputCoeffs = typename MatrixTraits<Out_Mean>::RowCoefficients;
      return zero_hessian<OutputCoeffs, In, Perturbations...>();
    }

  //protected:
    const JacobianFunction jacobian_fun;
  };


  //////////////////////////////////////////////
  //  Second-Order Linearized Transformation  //
  //////////////////////////////////////////////

  template<
    typename Function,
    typename JacobianFunction,
    typename HessianFunc>
  struct Transformation<Function, JacobianFunction, HessianFunc> : Transformation<Function, JacobianFunction>
  {
    using HessianFunction = HessianFunc;
    using Base = Transformation<Function, JacobianFunction>;

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

      return hessian_fun(std::forward<In>(in), internal::get_perturbation(std::forward<Perturbations>(ps))...);
    }

  protected:
    const HessianFunction hessian_fun;
  };


  /**
   * Deduction guide
   */

  template<typename Function, typename...TaylorDerivatives>
  Transformation(Function&&, TaylorDerivatives&&...) -> Transformation<Function, TaylorDerivatives...>;

  template<typename Function,
    std::enable_if_t<is_linearized_function_v<Function, 0> and not is_linearized_function_v<Function, 1>, int> = 0>
  Transformation(Function&&) -> Transformation<Function>;

  template<typename Function,
    std::enable_if_t<is_linearized_function_v<Function, 1> and not is_linearized_function_v<Function, 2>, int> = 0>
  Transformation(Function&&)
  -> Transformation<Function,
    decltype(is_linearized_function<Function, 1>::get_lambda(std::declval<Function>()))>;

  template<typename Function, std::enable_if_t<is_linearized_function_v<Function, 2>, int> = 0>
  Transformation(Function&&)
  -> Transformation<Function,
    decltype(is_linearized_function<Function, 1>::get_lambda(std::declval<Function>())),
    decltype(is_linearized_function<Function, 2>::get_lambda(std::declval<Function>()))>;


  //////////////////////
  //  Make functions  //
  //////////////////////

  /// Make a Transformation from a transformation function (and optionally one or more Taylor series derivatives).
  template<
    typename Function,
    typename...TaylorDerivatives,
    std::enable_if_t<not is_linearized_function_v<Function, 1>, int> = 0>
  auto make_Transformation(const Function& f, const TaylorDerivatives&...ds)
  {
    return Transformation<Function, TaylorDerivatives...>(f, ds...);
  };

  /// Make a transformation from a first-order linearized transformation defining a Jacobian function.
  /// Substitution failure if the transformation function is polymorphic.
  template<
    typename Function,
    std::enable_if_t<is_linearized_function_v<Function, 1> and not is_linearized_function_v<Function, 2>, int> = 0>
  auto make_Transformation(const Function& f)
  {
    return Transformation<Function,
      decltype(is_linearized_function<Function, 1>::get_lambda(std::declval<Function>()))>(f);
  };

  /// Make a transformation from a second-order linearized transformation defining Jacobian and Hessian functions.
  /// Substitution failure if the transformation function is polymorphic.
  template<
    typename Function,
    std::enable_if_t<is_linearized_function_v<Function, 2>, int> = 0>
  auto make_Transformation(const Function& f)
  {
    return Transformation<Function,
      decltype(is_linearized_function<Function, 1>::get_lambda(std::declval<Function>())),
      decltype(is_linearized_function<Function, 2>::get_lambda(std::declval<Function>()))>(f);
  };

}


#endif //OPENKALMAN_TRANSFORMATION_H
