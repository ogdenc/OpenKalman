/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref scalar_constant_operation.
 */

#ifndef OPENKALMAN_SCALAR_CONSTANT_OPERATION_HPP
#define OPENKALMAN_SCALAR_CONSTANT_OPERATION_HPP


namespace OpenKalman::internal
{
#ifndef __cpp_concepts
  template<typename Operation, typename...Ts> struct scalar_constant_operation;


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
      using value_type = std::decay_t<decltype(value)>;
      using type = scalar_constant_operation<Operation, Ts...>;
      static constexpr Likelihood status = (has_maybe_status<Ts>::value or ...) ? Likelihood::maybe : Likelihood::definitely;
      constexpr operator value_type() const noexcept { return value; }
      constexpr value_type operator()() const noexcept { return value; }
      constexpr auto operator+() { return static_cast<type&>(*this); }
      constexpr auto operator-() { return scalar_constant_operation {std::negate<>{}, static_cast<type&>(*this)}; }
    };


    // n-ary, not all arguments calculable at compile time
    template<typename Operation, typename...Ts>
    struct scalar_constant_operation_impl<Operation, std::enable_if_t<(scalar_constant<Ts> and ...) and
      std::is_invocable_v<const Operation&, decltype(get_scalar_constant_value(std::declval<const Ts&>()))...> and
      (not (scalar_constant<Ts, CompileTimeStatus::known> and ...) or
      not std::is_default_constructible_v<Operation> or not scalar_op_constexpr<Operation, void, Ts...>::value)>, Ts...>
    {
    private:

      template<typename U, typename = void>
      struct has_maybe_status : std::false_type {};

      template<typename U>
      struct has_maybe_status<U, std::enable_if_t<U::status == Likelihood::maybe>> : std::true_type {};

    public:

      explicit constexpr scalar_constant_operation_impl(const Operation& op, const Ts&...ts)
        : value {op(get_scalar_constant_value(ts)...)} {};

      using value_type = std::decay_t<decltype(std::declval<const Operation&>()(get_scalar_constant_value(std::declval<const Ts&>())...))>;
      using type = scalar_constant_operation<Operation, Ts...>;

      static constexpr Likelihood status = (has_maybe_status<Ts>::value or ...) ? Likelihood::maybe : Likelihood::definitely;
      constexpr operator value_type() const noexcept { return value; }
      constexpr value_type operator()() const noexcept { return value; }
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
    using value_type = std::decay_t<decltype(value)>;
    static constexpr Likelihood status = (has_maybe_status<Ts> or ...) ? Likelihood::maybe : Likelihood::definitely;
    using type = scalar_constant_operation;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
  };


  /**
   * \internal
   * \brief An operation involving some number of \ref scalar_constant values, which is not calculable at compile time.
   */
  template<typename Operation, scalar_constant...Ts> requires
    requires(const Operation& op, const Ts&...ts) { op(get_scalar_constant_value(ts)...); } and
    (not (scalar_constant<Ts, CompileTimeStatus::known> and ...) or
    not requires { typename std::bool_constant<(Operation{}(std::decay_t<Ts>::value...), true)>; })
  struct scalar_constant_operation<Operation, Ts...>
  {
  private:

    template<typename U>
    static constexpr bool has_maybe_status = requires { requires U::status == Likelihood::maybe; };

  public:

    explicit constexpr scalar_constant_operation(const Operation& op, const Ts&...ts)
      : value {op(get_scalar_constant_value(ts)...)} {};

    using value_type =
      std::decay_t<decltype(std::declval<const Operation&>()(get_scalar_constant_value(std::declval<const Ts&>())...))>;

    static constexpr Likelihood status = (has_maybe_status<Ts> or ...) ? Likelihood::maybe : Likelihood::definitely;
    using type = scalar_constant_operation;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }

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
  };
#endif


  //-- Deduction guides --//

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


} // namespace OpenKalman::internal

#endif //OPENKALMAN_SCALAR_CONSTANT_OPERATION_HPP
