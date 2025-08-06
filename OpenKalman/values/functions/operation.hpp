/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition of \ref values::operation, \ref values::operation_t and \ref values::consteval_operation.
 */

#ifndef OPENKALMAN_VALUES_OPERATION_HPP
#define OPENKALMAN_VALUES_OPERATION_HPP

#include "basics/basics.hpp"
#include "values/functions/to_number.hpp"
#include "values/traits/fixed_number_of.hpp"
#include "values/traits/number_type_of.hpp"

namespace OpenKalman::values
{
  /**
   * \brief An operation involving some number of \ref values::value "values".
   * \tparam Operation An operation taking <code>sizeof...(Args)</code> parameters
   * \tparam Args A set of \ref values::fixed types
   */
#ifdef __cpp_concepts
  template<typename Operation, fixed...Args> requires
    std::bool_constant<(stdcompat::invoke(Operation{}, fixed_number_of_v<Args>...), true)>::value
#else
  template<typename Operation, typename...Args>
#endif
  struct consteval_operation
  {
    constexpr consteval_operation() = default;
    explicit constexpr consteval_operation(const Operation&, const Args&...) {};
    static constexpr auto value = stdcompat::invoke(Operation{}, fixed_number_of_v<Args>...);
    using value_type = std::decay_t<decltype(value)>;
    using type = consteval_operation;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };


  /**
   * \brief Deduction guide.
   */
  template<typename Operation, typename...Args>
  explicit consteval_operation(const Operation&, const Args&...) -> consteval_operation<Operation, Args...>;


  namespace detail
  {
#ifdef __cpp_concepts
    template<typename Op, typename...Args>
#else
    template<typename Op, typename = void, typename...Args>
#endif
    struct operation_consteval_invocable_impl : std::false_type{};


#ifdef __cpp_concepts
    template<typename Op, typename...Args> requires requires { consteval_operation {std::declval<Op>(), std::declval<Args>()...}; }
    struct operation_consteval_invocable_impl<Op, Args...>
#else
    template<typename Op, typename...Args>
    struct operation_consteval_invocable_impl<Op, std::enable_if_t<std::bool_constant<(stdcompat::invoke(Op{}, fixed_number_of<Args>::value...), true)>::value>, Args...>
#endif
      : std::true_type{};


    template<typename Op, typename...Args>
#ifdef __cpp_concepts
    concept operation_consteval_invocable = operation_consteval_invocable_impl<Op, Args...>::value;
#else
    inline constexpr bool operation_consteval_invocable = operation_consteval_invocable_impl<Op, void, Args...>::value;
#endif
  }


  /**
   * \brief A potentially constant-evaluated operation involving some number of \ref values::value "values".
   * \details In this unspecialized case, the operation is not constant-evaluated.
   * \tparam Operation An operation taking <code>sizeof...(Args)</code> parameters
   * \tparam Args A set of \ref values::value types
   */
#ifdef __cpp_concepts
  template<typename Operation, value...Args> requires std::invocable<Operation&&, number_type_of_t<Args&&>...>
#else
  template<typename Operation, typename...Args>
#endif
  constexpr auto
  operation(Operation&& op, Args&&...args)
  {
    if constexpr ((... and fixed<Args>) and detail::operation_consteval_invocable<Operation, Args...>)
    {
      return consteval_operation {op, args...};
    }
    else
    {
      return stdcompat::invoke(std::forward<Operation>(op), to_number(std::forward<Args>(args))...);
    }
  }


  /**
   * \brief The resulting type from an \ref values::operation.
   */
  template<typename Operation, typename...Args>
  using operation_t = decltype(operation(std::declval<Operation&&>(), std::declval<Args&&>()...));

}

#endif
