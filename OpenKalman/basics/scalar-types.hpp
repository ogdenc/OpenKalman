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
 * \brief Concepts and traits for scalar types.
 */

#ifndef OPENKALMAN_SCALAR_TYPES_HPP
#define OPENKALMAN_SCALAR_TYPES_HPP

#include <complex>

namespace OpenKalman
{

  // --------------------------------------------- //
  //    ScalarTraits for arithmetic and complex    //
  // --------------------------------------------- //

  namespace interface
  {
  #ifdef __cpp_concepts
    template<typename T>
  #else
    template<typename T, typename = void>
  #endif
    struct ScalarTraits; // defined in forward-interface-traits.hpp


#ifdef __cpp_concepts
    template<typename T> requires std::is_arithmetic_v<T>
    struct ScalarTraits<T>
#else
    template<typename T>
    struct ScalarTraits<T, std::enable_if_t<std::is_arithmetic_v<T>>>
#endif
    {
      static constexpr bool is_complex = false;

      template<typename Re, typename Im>
      static constexpr decltype(auto) make_complex(Re&& re, Im&& im)
      {
        return std::complex<std::decay_t<T>>{std::forward<Re>(re), std::forward<Im>(im)};
      }

      template<typename Arg>
      static constexpr auto parts(Arg&& arg) { return std::forward_as_tuple(std::forward<Arg>(arg)); }

      template<typename Arg>
      static constexpr auto real_part(Arg&& arg) { return std::real(std::forward<Arg>(arg)); }

      template<typename Arg>
      static constexpr auto imaginary_part(Arg&& arg) { return std::imag(std::forward<Arg>(arg)); }

      template<typename Arg>
      static constexpr auto conj(Arg&& arg) { return std::forward<Arg>(arg); }

      template<typename Arg>
      static auto sin(Arg&& arg) { return std::sin(std::forward<Arg>(arg)); }

      template<typename Arg>
      static auto cos(Arg&& arg) { return std::cos(std::forward<Arg>(arg)); }

      template<typename Arg>
      static auto sqrt(Arg&& arg) { return std::sqrt(std::forward<Arg>(arg)); }

      template<typename Y, typename R>
      static auto asin2(Y&& y, R&& r)
      {
        // This is so that a zero-radius or faulty spherical coordinate has horizontal inclination:
        if (r == 0 or r < y or y < -r) return std::decay_t<decltype(std::asin(y/r))>{0};
        else return std::asin(y/r);
      }

      template<typename Y, typename X>
      static auto atan2(Y&& y, X&& x)
      {
        if constexpr (std::numeric_limits<std::decay_t<T>>::is_iec559) return std::atan2(y, x);
        else
        {
          using R = std::decay_t<decltype(std::atan2(y, x))>;
          if (y == 0) return std::copysign(std::signbit(x) ? R(0) : numbers::pi_v<R>, y);
          else if (not std::isfinite(y))
            return std::copysign(numbers::pi_v<R> * (std::isfinite(x) ? 0.5 : std::signbit(x) ? 0.25 : 0.75), y);
          else if (not std::isfinite(x))
          {
            if (std::signbit(x)) return std::copysign(numbers::pi_v<R>, y);
            else return std::copysign(R(0), y);
          }
          else if (x == 0) return std::copysign(numbers::pi_v<R>/2, y);
          else return std::atan2(y, x);
        }
      }
    };


    template<typename T>
    struct ScalarTraits<std::complex<T>>
    {
    private:

      using Scalar = std::conditional_t<std::is_integral_v<T>, double, T>;
      using Complex = std::complex<Scalar>;
      static constexpr auto pi = numbers::pi_v<Scalar>;

    public:

      static constexpr bool is_complex = true;

      template<typename Re, typename Im>
      static constexpr decltype(auto) make_complex(Re&& re, Im&& im)
      {
        return Complex{std::forward<Re>(re), std::forward<Im>(im)};
      }

      template<typename Arg>
      static constexpr std::tuple<Scalar, Scalar> parts(const Arg& arg) { return {std::real(arg), std::imag(arg)}; }

      template<typename Arg>
      static constexpr Scalar real_part(Arg&& arg) { return std::real(std::forward<Arg>(arg)); }

      template<typename Arg>
      static constexpr Scalar imaginary_part(Arg&& arg) { return std::imag(std::forward<Arg>(arg)); }

      template<typename Arg>
      static constexpr Complex conj(Arg&& arg)
      {
# ifdef __cpp_lib_constexpr_complex
          return std::conj(arg);
# else
          return make_complex(std::real(arg), -std::imag(arg));
# endif
      }

      template<typename Arg>
      static Complex sin(Arg&& arg) { return std::sin(std::forward<Arg>(arg)); }

      template<typename Arg>
      static Complex cos(Arg&& arg) { return std::cos(std::forward<Arg>(arg)); }

      template<typename Arg>
      static Complex sqrt(Arg&& arg) { return std::sqrt(std::forward<Arg>(arg)); }

      static Complex asin2(const Complex& y, const Complex& r)
      {
        if (r == Complex{0}) return Complex{0};
        else return std::asin(y/r);
      }

      static Complex atan2(const Complex& y, const Complex& x)
      {
        if (y == Complex{0})
        {
          return {std::copysign(std::signbit(std::real(x)) ? 0 : pi, std::real(y))};
        }
        else if (not std::isfinite(std::real(y)))
        {
          Scalar k = std::isfinite(std::real(x)) ? 0.5 : std::signbit(std::real(x)) ? 0.25 : 0.75;
          return {std::copysign(pi * k, std::real(y))};
        }
        else if (not std::isfinite(std::real(x)))
        {
          if (std::signbit(std::real(x))) return std::complex {std::copysign(pi, std::real(y))};
          else return {std::copysign(Scalar{0}, std::real(y))};
        }
        else if (std::real(x) > 0)
        {
          if constexpr (std::is_integral_v<T>)
            return std::atan(std::complex{std::real(y), std::imag(y)}/std::complex{std::real(x), std::imag(x)});
          else
            return std::atan(y/x);
        }
        else if (std::real(x) < 0)
        {
          if constexpr (std::is_integral_v<T>)
            return std::atan(std::complex{std::real(y), std::imag(y)}/std::complex{std::real(x), std::imag(x)}) +
              std::copysign(pi, std::real(y));
          else
            return std::atan(y/x) + std::copysign(pi, std::real(y));
        }
        else
          return {std::copysign(pi/2, std::real(y))}; // std::real(x) == 0
      }
    };

  } // namespace interface


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
   * \brief T is a std::complex.
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
      std::is_convertible<decltype(interface::ScalarTraits<T>::sin(std::declval<T>())), const T&>::value and
      std::is_convertible<decltype(interface::ScalarTraits<T>::cos(std::declval<T>())), const T&>::value and
      std::is_convertible<decltype(interface::ScalarTraits<T>::sqrt(std::declval<T>())), const T&>::value and
      std::is_convertible<decltype(interface::ScalarTraits<T>::asin2(std::declval<T>(), std::declval<T>())), const T&>::value and
      std::is_convertible<decltype(interface::ScalarTraits<T>::atan2(std::declval<T>(), std::declval<T>())), const T&>::value
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
      typename interface::ScalarTraits<std::decay_t<T>>;
      requires std::default_initializable<std::decay_t<T>>;
      {interface::ScalarTraits<std::decay_t<T>>::sin(t1)} -> std::convertible_to<const std::decay_t<T>&>;
      {interface::ScalarTraits<std::decay_t<T>>::cos(t1)} -> std::convertible_to<const std::decay_t<T>&>;
      {interface::ScalarTraits<std::decay_t<T>>::sqrt(t1)} -> std::convertible_to<const std::decay_t<T>&>;
      {interface::ScalarTraits<std::decay_t<T>>::asin2(t1, t2)} -> std::convertible_to<const std::decay_t<T>&>;
      {interface::ScalarTraits<std::decay_t<T>>::atan2(t1, t2)} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 + t2} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 - t2} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 * t2} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 / t2} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 == t2} -> std::convertible_to<const bool>;
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
      struct scalar_op_constexpr<Operation, std::enable_if_t<
        (Operation{}(std::decay_t<Ts>::value...), true)>, Ts...> : std::true_type {};


      template<typename Operation, typename = void, typename...Ts>
      struct scalar_constant_operation_impl {};


      // nullary
      template<typename Operation>
      struct scalar_constant_operation_impl<Operation, std::enable_if_t<scalar_op_constexpr<Operation>::value>>
      {
        constexpr scalar_constant_operation_impl() = default;
        explicit constexpr scalar_constant_operation_impl(const Operation&) {};
        static constexpr auto value = Operation{}();
        static constexpr auto status = Likelihood::definitely;
        using value_type = std::decay_t<decltype(value)>;
        constexpr operator value_type() const noexcept { return value; }
        constexpr value_type operator()() const noexcept { return value; }
      };


      // n-ary, all arguments calculable at compile time
      template<typename Operation, typename T, typename...Ts>
      struct scalar_constant_operation_impl<Operation, std::enable_if_t<
        (scalar_constant<T, CompileTimeStatus::known> and ... and scalar_constant<Ts, CompileTimeStatus::known>) and
        scalar_op_constexpr<Operation, void, T, Ts...>::value>, T, Ts...>
      {
      private:

        template<typename U, typename = void>
        struct constant_status : std::integral_constant<Likelihood, Likelihood::definitely> {};

        template<typename U>
        struct constant_status<U, std::enable_if_t<std::is_convertible_v<decltype(U::status), Likelihood>>>
          : std::integral_constant<Likelihood, U::status> {};

      public:

        constexpr scalar_constant_operation_impl() = default;
        constexpr scalar_constant_operation_impl(const Operation&, const T&, const Ts&...) {};
        static constexpr auto value = Operation{}(std::decay_t<T>::value, std::decay_t<Ts>::value...);
        static constexpr auto status = (constant_status<T>::value and ... and constant_status<Ts>::value);
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
        template<typename...Args>
        scalar_constant_operation_impl(const Operation& op, Args&&...args) : value {op(get_scalar_constant_value(std::forward<Args>(args))...)} {};

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
    struct scalar_constant_operation;


    /**
     * \internal
     * \brief A constexpr nullary operation.
     */
    template<typename Operation> requires (Operation{}(), true)
    struct scalar_constant_operation<Operation>
    {
      constexpr scalar_constant_operation() = default;
      explicit constexpr scalar_constant_operation(const Operation&) {};
      static constexpr auto value = Operation{}();
      static constexpr auto status = Likelihood::definitely;
      using value_type = std::decay_t<decltype(value)>;
      using type = scalar_constant_operation;
      constexpr operator value_type() const noexcept { return value; }
      constexpr value_type operator()() const noexcept { return value; }
    };


    /**
     * \internal
     * \overload
     * \brief An operation involving some number of compile-time \ref scalar_constant values.
     */
    template<typename Operation, scalar_constant<CompileTimeStatus::known> T, scalar_constant<CompileTimeStatus::known>...Ts>
    requires requires { requires (Operation{}(std::decay_t<T>::value, std::decay_t<Ts>::value...), true); }
    struct scalar_constant_operation<Operation, T, Ts...>
    {
    private:

      template<typename U, typename = void>
      struct constant_status : std::integral_constant<Likelihood, Likelihood::definitely> {};

      template<typename U>
      struct constant_status<U, std::enable_if_t<std::is_convertible_v<decltype(U::status), Likelihood>>>
        : std::integral_constant<Likelihood, U::status> {};

    public:

      constexpr scalar_constant_operation() = default;
      constexpr scalar_constant_operation(const Operation&, const T&, const Ts&...) {};
      static constexpr auto value = Operation{}(std::decay_t<T>::value, std::decay_t<Ts>::value...);
      static constexpr auto status = (constant_status<T>::value and ... and constant_status<Ts>::value);
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
      scalar_constant_operation(const Operation& op, scalar_constant auto&&...args)
        : value {op(get_scalar_constant_value(std::forward<decltype(args)>(args))...)} {};

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
    scalar_constant_operation(const Operation&, const Ts&...) -> scalar_constant_operation<Operation, Ts...>;


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
