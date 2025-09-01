/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TRANSFORMATION_HPP
#define OPENKALMAN_TRANSFORMATION_HPP

#include <type_traits>

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  /**
   * \brief A tests from one single-column vector to another.
   * \details Models a tests (linear or nonlinear) from one single-column vector to another.
   * The tests takes an input vector, and optionally one or more perturbation terms. These can be
   * associated with noise, or translation, etc. The perturbation terms can either be constant single-column
   * vectors, or statistical distributions (in which case, the perturbation will be stochastic).
   * \tparam Function The tests function, in the following exemplary form:
   * (Mean<InputCoefficients,...>, Mean<OutputCoefficients,...>, ...) -> Mean<OutputCoefficients,...>.
   * The first term is the input, the next term(s) represent perturbation(s), and the final term is the output.
   * \tparam TaylorDerivatives Optional Taylor-series derivative functions, including the Jacobian and Hessian
   * for the input and each perturbation.
   */
  template<typename Function, typename ... TaylorDerivatives>
  struct Transformation;


  namespace internal
  {
#ifdef __cpp_concepts
    template<typename Function, std::size_t order, typename...TaylorDerivatives> requires
      (order <= sizeof...(TaylorDerivatives))
    struct is_linearized_function<Transformation<Function, TaylorDerivatives...>, order> : std::true_type {};
#else
    template<typename Function, std::size_t order, typename...TaylorDerivatives>
    struct is_linearized_function<Transformation<Function, TaylorDerivatives...>, order, std::enable_if_t<
      (order <= sizeof...(TaylorDerivatives))>> : std::true_type {};
#endif

  }


  // ------------------------------  //
  //  Non-linearized Transformation  //
  // ------------------------------  //

  template<typename Function>
  struct Transformation<Function>
  {

    /// Default constructor.
    Transformation()
      : function {Function()} {}


    /// Constructor from a tests function.
    template<typename F>
    Transformation(F&& f) : function {std::forward<F>(f)} {}


    /// Applies the tests.
#ifdef __cpp_concepts
    template<transformation_input In, perturbation ... Perturbations>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<transformation_input<In> and
      (perturbation<Perturbations> and ...), int> = 0>
#endif
    auto operator()(In&& in, Perturbations&& ... ps) const
    {
      return function(std::forward<In>(in), oin::get_perturbation(std::forward<Perturbations>(ps))...);
    }

  private:

    const Function function; ///< The tests function.

  };


  // --------------------------------------  //
  //  First-Order Linearized Transformation  //
  // --------------------------------------  //

  template<typename Function, typename JacobianFunction>
  struct Transformation<Function, JacobianFunction> : Transformation<Function>
  {

  private:

    using Base = Transformation<Function>;

  protected:

    static auto default_Jacobian(const Function& f)
    {
      if constexpr (linearized_function<Function, 1>) return oin::get_Taylor_term<1>(f);
      else return JacobianFunction();
    }

  public:

    /// Default constructor.
    Transformation()
      : Base {Function()}, jacobian_fun {JacobianFunction()} {}


    /// Constructs tests using a tests function.
    template<typename F>
    Transformation(const F& f)
      : Base {f}, jacobian_fun {default_Jacobian(f)} {}


    /// Constructs tests using a tests function and a Jacobian function.
    template<typename F, typename J>
    Transformation(const F& f, const J& j)
      : Base {f}, jacobian_fun {j} {}


    /// Returns a tuple of the Jacobians for the input and each perturbation term.
#ifdef __cpp_concepts
    template<transformation_input In, perturbation ... Perturbations>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<transformation_input<In> and
      (perturbation<Perturbations> and ...), int> = 0>
#endif
    auto jacobian(In&& in, Perturbations&& ... ps) const
    {
      return jacobian_fun(std::forward<In>(in), oin::get_perturbation(std::forward<Perturbations>(ps))...);
    }

  private:

    const JacobianFunction jacobian_fun;
  };


  // ---------------------------------------- //
  //  Second-Order Linearized Transformation  //
  // ---------------------------------------- //

  template<
    typename Function,
    typename JacobianFunction,
    typename HessianFunction>
  struct Transformation<Function, JacobianFunction, HessianFunction> : Transformation<Function, JacobianFunction>
  {

  private:

    using Base = Transformation<Function, JacobianFunction>;

  protected:

    static auto default_Hessian(const Function& f)
    {
      if constexpr (linearized_function<Function, 2>) return oin::get_Taylor_term<2>(f);
      else return HessianFunction();
    }

  public:

    /// Constructs tests using a tests function and a Jacobian and Hessian functions.
    /// Default constructor.
    Transformation()
      : Base(), hessian_fun(HessianFunction()) {}


    /// Constructs tests using a tests function.
    template<typename F>
    Transformation(const F& f)
      : Base {f}, hessian_fun {default_Hessian(f)} {}


    /// Constructs tests using a tests function and a Jacobian function.
    template<typename F, typename J>
    Transformation(const F& f, const JacobianFunction& j)
      : Base(f, j), hessian_fun(default_Hessian(f)) {}


    /// Constructs tests using a tests function and a Jacobian function.
    template<typename F, typename J, typename H>
    Transformation(const F& f, const J& j, const H& h)
      : Base(f, j), hessian_fun(h) {}


    /// Returns a tuple of Hessian matrices for the input and each perturbation term.
#ifdef __cpp_concepts
    template<transformation_input In, perturbation ... Perturbations>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<transformation_input<In> and
      (perturbation<Perturbations> and ...), int> = 0>
#endif
    auto hessian(In&& in, Perturbations&& ... ps) const
    {
      return hessian_fun(std::forward<In>(in), oin::get_perturbation(std::forward<Perturbations>(ps))...);
    }

  private:

    const HessianFunction hessian_fun;

  };


  // ------------------ //
  //  Deduction guides  //
  // ------------------ //

  template<typename Function, typename ... TaylorDerivatives>
  Transformation(Function&&, TaylorDerivatives&& ...)
    -> Transformation<std::decay_t<Function>, std::decay_t<TaylorDerivatives> ...>;


#ifdef __cpp_concepts
  template<linearized_function<0> Function> requires (not linearized_function<Function, 1>)
#else
  template<typename Function, std::enable_if_t<
    linearized_function<Function, 0> and (not linearized_function<Function, 1>), int> = 0>
#endif
  Transformation(Function&&) -> Transformation<std::decay_t<Function>>;


#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS_2
  template<linearized_function<1> Function> requires (not linearized_function<Function, 2>)
  // \todo Unlike SFINAE version, this incorrectly matches linearized_function<0> in both GCC 10.1.0 and clang 10.0.0:
#else
  template<typename Function, std::enable_if_t<
    linearized_function<Function, 1> and (not linearized_function<Function, 2>), int> = 0>
#endif
  Transformation(Function&&) -> Transformation<std::decay_t<Function>,
    std::decay_t<decltype(oin::get_Taylor_term<1>(std::declval<Function>()))>>;


#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS_2
  template<linearized_function<2> Function>
  // \todo Unlike SFINAE version, this incorrectly matches linearized_function<0> in both GCC 10.1.0 and clang 10.0.0:
#else
  template<typename Function, std::enable_if_t<linearized_function<Function, 2>, int> = 0>
#endif
  Transformation(Function&&) -> Transformation<std::decay_t<Function>,
    std::decay_t<decltype(oin::get_Taylor_term<1>(std::declval<Function>()))>,
    std::decay_t<decltype(oin::get_Taylor_term<2>(std::declval<Function>()))>>;


}


#endif
