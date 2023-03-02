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

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_scalar_type : std::false_type {};

    template<typename T>
    struct is_scalar_type<T, std::enable_if_t<
      std::is_default_constructible_v<T> and
      std::is_convertible_v<decltype(std::declval<T>() + std::declval<T>()), const std::decay_t<T>&> and
      std::is_convertible_v<decltype(std::declval<T>() - std::declval<T>()), const std::decay_t<T>&> and
      std::is_convertible_v<decltype(std::declval<T>() * std::declval<T>()), const std::decay_t<T>&> and
      std::is_convertible_v<decltype(std::declval<T>() / std::declval<T>()), const std::decay_t<T>&> and
      std::is_convertible_v<decltype(std::declval<T>() == std::declval<T>()), const bool>
      >>: std::true_type {};
  }
#endif


  /**
   * \brief T is a scalar type.
   * \details T can be any arithmetic, complex, or custom scalar type in which certain traits in
   * interface::ScalarTraits are defined and typical math operations (+, -, *, /, and ==) are also defined.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept scalar_type = std::is_arithmetic_v<std::decay_t<T>> or complex_number<T> or
    requires(std::decay_t<T> t1, std::decay_t<T> t2) {
      requires std::default_initializable<std::decay_t<T>>;
      {t1 + t2} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 - t2} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 * t2} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 / t2} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 == t2} -> std::convertible_to<const bool>;
      {interface::ScalarTraits<std::decay_t<T>>::make_complex(real(t1), imag(t1))} -> std::convertible_to<const std::decay_t<T>&>;
    };
#else
  constexpr bool scalar_type =
    std::is_arithmetic_v<std::decay_t<T>> or complex_number<T> or detail::is_scalar_type<std::decay_t<T>>::value;
#endif


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
      struct is_compile_time_scalar_constant : std::false_type {};

      template<typename T>
      struct is_compile_time_scalar_constant<T, std::enable_if_t<std::is_default_constructible_v<std::decay_t<T>> and
        scalar_type<decltype(std::decay_t<T>::value)>>>
        : std::bool_constant<std::is_convertible_v<T, const decltype(std::decay_t<T>::value)>> {};
    }
#endif


    /**
     * \brief T is a scalar constant known at compile time
     */
    template<typename T>
#ifdef __cpp_concepts
    concept compile_time_scalar_constant = std::default_initializable<std::decay_t<T>> and
      requires {
        {std::decay_t<T>::value} -> scalar_type;
        static_cast<const decltype(std::decay_t<T>::value)>(std::declval<T>());
      };
#else
    constexpr bool compile_time_scalar_constant = detail::is_compile_time_scalar_constant<T>::value;
#endif


#ifndef __cpp_concepts
    namespace detail
    {
      template<typename T, typename = void>
      struct is_runtime_scalar : std::false_type {};

      template<typename T>
      struct is_runtime_scalar<T, std::enable_if_t<(scalar_type<typename std::invoke_result<std::decay_t<T>>::type>)>>
        : std::true_type {};
    }
#endif

    /**
     * \brief T is a scalar constant known at runtime
     */
    template<typename T>
#ifdef __cpp_concepts
    concept runtime_scalar_constant = (not compile_time_scalar_constant<T>) and
      (scalar_type<T> or requires(std::decay_t<T> t){ {t()} -> scalar_type; });
#else
    constexpr bool runtime_scalar_constant = (not compile_time_scalar_constant<T>) and
      (scalar_type<T> or detail::is_runtime_scalar<T>::value);
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
    if constexpr (internal::compile_time_scalar_constant<Arg>)
      return std::decay_t<Arg>::value;
    else if constexpr (scalar_type<Arg>)
      return std::forward<Arg>(arg);
    else
      return std::forward<Arg>(arg)();
  }


  // ---------------------- //
  //    real_axis_number    //
  // ---------------------- //

  namespace detail
  {
    template<typename T>
    constexpr bool imaginary_part_is_zero()
    {
      using std::imag;
      constexpr auto v = std::decay_t<T>::value;
      if constexpr (complex_number<decltype(v)>) return imag(v) == 0;
      else return true;
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
      struct scalar_op_constexpr<Operation, std::enable_if_t<(Operation{}(Ts::value...), true)>, Ts...> : std::true_type {};


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
      requires { requires (Operation{}(std::decay_t<Ts>::value...), true); }
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
      (not requires { requires (Operation{}(std::decay_t<Ts>::value...), true); })
    struct scalar_constant_operation<Operation, Ts...>
    {
      explicit constexpr scalar_constant_operation(const Operation& op, const Ts&...ts)
        : value {op(get_scalar_constant_value(ts)...)} {};

      using value_type =
        std::decay_t<decltype(std::declval<const Operation&>()(get_scalar_constant_value(std::declval<const Ts&>())...))>;

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
