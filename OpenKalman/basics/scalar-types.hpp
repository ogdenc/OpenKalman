/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Concepts and related facilities for scalar types.
 */

#ifndef OPENKALMAN_SCALAR_TYPES_HPP
#define OPENKALMAN_SCALAR_TYPES_HPP

#include <complex>

namespace OpenKalman
{
  // -------------------- //
  //    complex_number    //
  // -------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_complex_number : std::false_type {};


    template<typename T>
    struct is_complex_number<T, std::enable_if_t<interface::ScalarTraits<std::decay_t<T>>::is_complex>> : std::true_type {};
  }
#endif


  /**
   * \brief T is std::complex or a custom complex type.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept complex_number = interface::ScalarTraits<std::decay_t<T>>::is_complex;
#else
  constexpr bool complex_number = detail::is_complex_number<std::decay_t<T>>::value;
#endif


  // --------------- //
  //   scalar_type   //
  // --------------- //

  /**
   * \brief T is a scalar type.
   * \details T can be any arithmetic, complex, or custom scalar type in which certain traits in
   * interface::ScalarTraits are defined and typical math operations (+, -, *, /, and ==) are also defined.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept scalar_type =
#else
  constexpr bool scalar_type =
#endif
    std::numeric_limits<std::decay_t<T>>::is_specialized or complex_number<T>;


  // ------------------------ //
  //   floating_scalar_type   //
  // ------------------------ //

  /**
   * \brief T is a floating-point scalar type.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept floating_scalar_type =
#else
  constexpr bool floating_scalar_type =
#endif
    scalar_type<T> and not std::is_integral_v<std::decay_t<T>> and not complex_number<T>;


  // --------------------- //
  //    scalar_constant    //
  // --------------------- //

  namespace internal
  {
#ifndef __cpp_concepts
    namespace detail
    {
      template<typename T, typename = void>
      struct has_value_member : std::false_type {};

      template<typename T>
      struct has_value_member<T, std::enable_if_t<scalar_type<decltype(T::value)>>> : std::true_type {};

      template<typename T, typename = void>
      struct call_result_is_scalar : std::false_type {};

      template<typename T>
      struct call_result_is_scalar<T, std::void_t<std::bool_constant<(T{}(), true)>>>
        : std::bool_constant<scalar_type<decltype(T{}())>> {};

      template<typename T, typename = void>
      struct convertible_to_value_type : std::false_type {};

      template<typename T>
      struct convertible_to_value_type<T, std::void_t<std::bool_constant<(typename T::value_type {T{}}, true)>>>
        : std::bool_constant<scalar_type<decltype(typename T::value_type {T{}})>> {};
    }
#endif


    /**
     * \brief T is a scalar constant known at compile time
     */
    template<typename T>
#ifdef __cpp_concepts
    concept compile_time_scalar_constant = std::default_initializable<std::decay_t<T>> and
      ( requires { {std::decay_t<T>::value} -> scalar_type; } or
        requires {
          {std::decay_t<T>{}()} -> scalar_type;
          typename std::bool_constant<(std::decay_t<T>{}(), true)>; } or
        requires {
          {typename std::decay_t<T>::value_type {std::decay_t<T>{}}} -> scalar_type;
          typename std::bool_constant<(typename std::decay_t<T>::value_type {std::decay_t<T>{}}, true)>; });
#else
    constexpr bool compile_time_scalar_constant = std::is_default_constructible_v<std::decay_t<T>> and
      (detail::has_value_member<std::decay_t<T>>::value or detail::call_result_is_scalar<std::decay_t<T>>::value or
        detail::convertible_to_value_type<std::decay_t<T>>::value);
#endif


#ifndef __cpp_concepts
    namespace detail
    {
      template<typename T, typename = void>
      struct is_runtime_scalar : std::false_type {};

      template<typename T>
      struct is_runtime_scalar<T, std::enable_if_t<scalar_type<typename std::invoke_result<T>::type>>>
        : std::true_type {};

      template<typename T, typename = void>
      struct is_runtime_convertible_scalar : std::false_type {};

      template<typename T>
      struct is_runtime_convertible_scalar<T, std::enable_if_t<scalar_type<decltype(typename T::value_type {std::declval<T>()})>>>
        : std::true_type {};
    }
#endif

    /**
     * \brief T is a scalar constant known at runtime
     */
    template<typename T>
#ifdef __cpp_concepts
    concept runtime_scalar_constant = (not compile_time_scalar_constant<T>) and
      (scalar_type<T> or requires(std::decay_t<T> t){ {t()} -> scalar_type; } or
        requires(std::decay_t<T> t){ {typename std::decay_t<T>::value_type {t}} -> scalar_type; });
#else
    constexpr bool runtime_scalar_constant = (not compile_time_scalar_constant<T>) and
      (scalar_type<T> or detail::is_runtime_scalar<std::decay_t<T>>::value or
      detail::is_runtime_convertible_scalar<std::decay_t<T>>::value);
#endif
  } // namespace internal


  /**
   * \brief T is a scalar constant
   * \tparam c Whether the constant is known or unknown at compile time.
   */
  template<typename T, CompileTimeStatus c = CompileTimeStatus::any>
#ifdef __cpp_concepts
  concept scalar_constant =
#else
  constexpr bool scalar_constant =
#endif
    (c != CompileTimeStatus::known or internal::compile_time_scalar_constant<T>) and
    (c != CompileTimeStatus::unknown or internal::runtime_scalar_constant<T>) and
    (c != CompileTimeStatus::any or internal::compile_time_scalar_constant<T> or internal::runtime_scalar_constant<T>);


  // ----------------------------- //
  //   get_scalar_constant_value   //
  // ----------------------------- //

  /**
   * \brief Get the \ref scalar_type of a |ref scalar_constant
   */
#ifdef __cpp_concepts
  template<scalar_constant Arg>
  constexpr scalar_type decltype(auto) get_scalar_constant_value(Arg&& arg)
#else
  template<typename Arg, std::enable_if_t<scalar_constant<Arg>, int> = 0>
  constexpr decltype(auto) get_scalar_constant_value(Arg&& arg)
#endif
  {
    using T = std::decay_t<Arg>;
#ifdef __cpp_concepts
    if constexpr (requires { {T::value} -> scalar_type; }) return T::value;
    else if constexpr (requires { {arg()} -> scalar_type; }) return std::forward<Arg>(arg)();
    else if constexpr (requires { {typename T::value_type {arg}} -> scalar_type; }) return typename T::value_type {std::forward<Arg>(arg)};
    else if constexpr (scalar_type<Arg>) return std::forward<Arg>(arg);
#else
    if constexpr (internal::detail::has_value_member<T>::value) return T::value;
    else if constexpr (internal::detail::call_result_is_scalar<T>::value or
      internal::detail::is_runtime_scalar<T>::value) return std::forward<Arg>(arg)();
    else if constexpr (internal::detail::convertible_to_value_type<T>::value or
      internal::detail::is_runtime_convertible_scalar<T>::value) return typename T::value_type {std::forward<Arg>(arg)};
    else if constexpr (scalar_type<Arg>) return std::forward<Arg>(arg);
    else return std::forward<Arg>(arg)();
#endif
  }


  // ---------------------- //
  //    real_axis_number    //
  // ---------------------- //

  namespace detail
  {
    template<typename T>
    constexpr bool imaginary_part_is_zero() { return true; }

#ifdef __cpp_concepts
    template<complex_number T>
#else
    template<typename T, typename = std::enable_if_t<complex_number<T> and scalar_constant<T, CompileTimeStatus::known>>>
#endif
    constexpr bool imaginary_part_is_zero()
    {
      using std::imag;
      return imag(std::decay_t<T>::value) == 0;
    }
  }


  /**
   * \brief T is either not a \ref complex_number or its imaginary component is 0.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept real_axis_number =
#else
  constexpr bool real_axis_number =
#endif
    scalar_constant<T, CompileTimeStatus::known> and detail::imaginary_part_is_zero<std::decay_t<T>>();


  namespace internal
  {
    // --------------------------- //
    //  scalar_constant_operation  //
    // --------------------------- //

#ifndef __cpp_concepts
    namespace detail
    {
      template<typename Operation, typename = void, typename...Ts>
      struct scalar_op_constexpr : std::false_type {};

      template<typename Operation, typename...Ts>
      struct scalar_op_constexpr<Operation, std::void_t<
        std::bool_constant<(Operation{}(std::decay_t<Ts>::value...), true)>>, Ts...> : std::true_type {};


      template<typename Operation, typename = void, typename...Ts>
      struct scalar_constant_operation_impl
      {
        explicit constexpr scalar_constant_operation_impl(const Operation&, const Ts&...) {};
      };


      // n-ary, all arguments calculable at compile time
      template<typename Operation, typename...Ts>
      struct scalar_constant_operation_impl<Operation, std::enable_if_t<
        (scalar_constant<Ts, CompileTimeStatus::known> and ...) and scalar_op_constexpr<Operation, void, Ts...>::value>, Ts...>
      {
      private:

        template<typename U, typename = void>
        struct has_maybe_status : std::false_type {};

        template<typename U>
        struct has_maybe_status<U, std::enable_if_t<U::status == Likelihood::maybe>> : std::true_type {};

      public:

        constexpr scalar_constant_operation_impl() = default;
        explicit constexpr scalar_constant_operation_impl(const Operation&, const Ts&...) {};
        static constexpr auto value = Operation{}(Ts::value...);
        static constexpr Likelihood status = (has_maybe_status<Ts>::value or ...) ? Likelihood::maybe : Likelihood::definitely;
        using value_type = std::decay_t<decltype(value)>;
        constexpr operator value_type() const noexcept { return value; }
        constexpr value_type operator()() const noexcept { return value; }
      };


      // n-ary, not all arguments calculable at compile time
      template<typename Operation, typename...Ts>
      struct scalar_constant_operation_impl<Operation, std::enable_if_t<(scalar_constant<Ts> and ...) and
        (not (scalar_constant<Ts, CompileTimeStatus::known> and ...) or
        not std::is_default_constructible_v<Operation> or not scalar_op_constexpr<Operation, void, Ts...>::value)>, Ts...>
      {
        explicit constexpr scalar_constant_operation_impl(const Operation& op, const Ts&...ts)
          : value {op(get_scalar_constant_value(ts)...)} {};

        using value_type = std::decay_t<decltype(std::declval<const Operation&>()(get_scalar_constant_value(std::declval<const Ts&>())...))>;

        operator value_type() const noexcept { return value; }
        value_type operator()() const noexcept { return value; }
      private:
        value_type value;
      };
    }
#endif


    /**
     * \internal
     * \brief An operation involving some number of \ref scalar_constant values.
     * \tparam Operation An operation taking <code>sizeof...(Ts)</code> parameters
     * \tparam Ts A set of \ref scalar_constant types
     */
#ifdef __cpp_concepts
    template<typename Operation, typename...Ts>
    struct scalar_constant_operation
    {
      explicit constexpr scalar_constant_operation(const Operation&, const Ts&...) {};
    };


    /**
     * \internal
     * \overload
     * \brief An operation involving some number of compile-time \ref scalar_constant values.
     */
    template<typename Operation, scalar_constant<CompileTimeStatus::known>...Ts> requires
      requires { typename std::bool_constant<(Operation{}(std::decay_t<Ts>::value...), true)>; }
    struct scalar_constant_operation<Operation, Ts...>
    {
    private:

      template<typename U>
      static constexpr bool has_maybe_status = requires { requires U::status == Likelihood::maybe; };

    public:

      constexpr scalar_constant_operation() = default;
      explicit constexpr scalar_constant_operation(const Operation&, const Ts&...) {};
      static constexpr auto value = Operation{}(Ts::value...);
      static constexpr Likelihood status = (has_maybe_status<Ts> or ...) ? Likelihood::maybe : Likelihood::definitely;
      using value_type = std::decay_t<decltype(value)>;
      using type = scalar_constant_operation;
      constexpr operator value_type() const noexcept { return value; }
      constexpr value_type operator()() const noexcept { return value; }
    };


    /**
     * \internal
     * \overload
     * \brief An operation involving some number of \ref scalar_constant values, which is not calculatable at compile time.
     */
    template<typename Operation, scalar_constant...Ts> requires
      (not (scalar_constant<Ts, CompileTimeStatus::known> and ...)) or
      (not requires { typename std::bool_constant<(Operation{}(std::decay_t<Ts>::value...), true)>; })
    struct scalar_constant_operation<Operation, Ts...>
    {
      explicit constexpr scalar_constant_operation(const Operation& op, const Ts&...ts)
        : value {op(get_scalar_constant_value(ts)...)} {};

      using value_type =
        std::decay_t<decltype(std::declval<Operation>()(get_scalar_constant_value(std::declval<const Ts&>())...))>;

      using type = scalar_constant_operation;
      operator value_type() const noexcept { return value; }
      value_type operator()() const noexcept { return value; }

    private:

      value_type value;
    };
#else
    template<typename Operation, typename...Ts>
    struct scalar_constant_operation : detail::scalar_constant_operation_impl<Operation, void, Ts...>
    {
    private:
      using Base = detail::scalar_constant_operation_impl<Operation, void, Ts...>;
    public:
      using Base::Base;
      using type = scalar_constant_operation;
    };
#endif


    /**
     * \internal
     * \brief Deduction guide.
     */
    template<typename Operation, typename...Ts>
    explicit scalar_constant_operation(const Operation&, const Ts&...) -> scalar_constant_operation<Operation, Ts...>;


    /**
     * \internal
     * \brief Helper template for \ref scalar_constant_operation.
     */
#ifdef __cpp_concepts
    template<typename Operation, scalar_constant<CompileTimeStatus::known>...Ts>
#else
    template<typename Operation, typename...Ts>
#endif
    constexpr auto scalar_constant_operation_v = scalar_constant_operation<Operation, Ts...>::value;

  } // namespace internal

} // namespace OpenKalman

#endif //OPENKALMAN_SCALAR_TYPES_HPP
