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
 * \brief Functions for solving linear equations.
 */

#ifndef OPENKALMAN_SOLVE_HPP
#define OPENKALMAN_SOLVE_HPP

namespace OpenKalman
{
  namespace detail
  {
    template<typename A, typename B>
    void solve_check_A_and_B_rows_match(const A& a, const B& b)
    {
      if (get_vector_space_descriptor<0>(a) != get_vector_space_descriptor<0>(b))
        throw std::domain_error {"The rows of the two operands of the solve function must be the same, but instead "
          "the first operand has " + std::to_string(get_index_dimension_of<0>(a)) + " rows and the second operand has " +
          std::to_string(get_index_dimension_of<0>(b)) + " rows"};
    }
  }


  /**
   * \brief Solve the equation AX = B for X, which may or may not be a unique solution.
   * \details The interface to the relevant linear algebra library determines what happens if A is not invertible.
   * \tparam must_be_unique Determines whether the function throws an exception if the solution X is non-unique
   * (e.g., if the equation is under-determined)
   * \tparam must_be_exact Determines whether the function throws an exception if it cannot return an exact solution,
   * such as if the equation is over-determined. If <code>false<code>, then the function will return an estimate
   * instead of throwing an exception.
   * \tparam A The matrix A in the equation AX = B
   * \tparam B The matrix B in the equation AX = B
   * \return The unique solution X of the equation AX = B. If <code>must_be_unique</code>, then the function can return
   * any valid solution for X. In particular, if <code>must_be_unique</code>, the function has the following behavior:
   * - If A is a \ref zero_matrix, then the result X will also be a \ref zero_matrix
   */
  #ifdef __cpp_concepts
  template<bool must_be_unique = false, bool must_be_exact = false, typename A, typename B> requires
    (not zero_matrix<A> or not zero_matrix<B> or not must_be_unique) and
    (not zero_matrix<A> or not (constant_matrix<B> or constant_diagonal_matrix<B>) or zero_matrix<B> or not must_be_exact) and
    (not constant_matrix<A> or not constant_diagonal_matrix<B> or has_dynamic_dimensions<A> or
      (index_dimension_of_v<A, 0> <= index_dimension_of_v<A, 1> and index_dimension_of_v<B, 0> <= index_dimension_of_v<A, 1>) or
      (index_dimension_of_v<A, 0> == 1 and index_dimension_of_v<B, 0> == 1) or not must_be_exact)
  constexpr compatible_with_vector_space_descriptors<vector_space_descriptor_of_t<A, 1>, vector_space_descriptor_of_t<B, 1>> auto
  #else
  template<bool must_be_unique = false, bool must_be_exact = false, typename A, typename B, std::enable_if_t<
    (not zero_matrix<A> or not zero_matrix<B> or not must_be_unique) and
    (not zero_matrix<A> or not (constant_matrix<B> or constant_diagonal_matrix<B>) or zero_matrix<B> or not must_be_exact) and
    (not constant_matrix<A> or not constant_diagonal_matrix<B> or has_dynamic_dimensions<A> or
      (index_dimension_of_v<A, 0> <= index_dimension_of_v<A, 1> and index_dimension_of_v<B, 0> <= index_dimension_of_v<A, 1>) or
      (index_dimension_of_v<A, 0> == 1 and index_dimension_of_v<B, 0> == 1) or not must_be_exact), int> = 0>
  constexpr auto
  #endif
  solve(A&& a, B&& b)
  {
    static_assert(dynamic_dimension<A, 0> or dynamic_dimension<B, 0> or index_dimension_of_v<A, 0> == index_dimension_of_v<B, 0>,
      "The rows of two operands of the solve function must be the same.");

    if constexpr (zero_matrix<B>)
    {
      if constexpr (dynamic_dimension<A, 0> or dynamic_dimension<B, 0>) detail::solve_check_A_and_B_rows_match(a, b);

      if constexpr (must_be_unique and not constant_matrix<A> and not constant_diagonal_matrix<A>)
      {
        // \todo the predicate should be simpler, such as (a == to_native_matrix<A>(make_zero_matrix_like(a))), but this causes a failure in Eigen.
        if (reduce([](auto c1, auto c2) { if (c1 == 0 and c2 == 0) return 0; else return 1; }, a) == 0)
          throw std::runtime_error {"solve function requires a unique solution, "
            "but because operands A and B are both zero matrices, result X may take on any value"};
        else
          return make_zero_matrix_like<B>(get_vector_space_descriptor<1>(a), get_vector_space_descriptor<1>(b));
      }
      else
        return make_zero_matrix_like<B>(get_vector_space_descriptor<1>(a), get_vector_space_descriptor<1>(b));
    }
    else if constexpr (zero_matrix<A>) //< This will be a non-exact solution unless b is zero.
    {
      if constexpr (dynamic_dimension<A, 0> or dynamic_dimension<B, 0>) detail::solve_check_A_and_B_rows_match(a, b);
        return make_zero_matrix_like<B>(get_vector_space_descriptor<1>(a), get_vector_space_descriptor<1>(b));
    }
    else if constexpr (constant_diagonal_matrix<A>)
    {
      if constexpr (dynamic_dimension<A, 0> or dynamic_dimension<B, 0>) detail::solve_check_A_and_B_rows_match(a, b);
      if constexpr (identity_matrix<A>)
        return std::forward<B>(b);
      else
        return make_self_contained(std::forward<B>(b) / constant_diagonal_coefficient{a}());
    }
    else if constexpr (constant_matrix<A>)
    {
      if constexpr ((index_dimension_of_v<A, 0> == 1 or index_dimension_of_v<B, 0> == 1) and index_dimension_of_v<A, 1> == 1)
      {
        if constexpr (dynamic_dimension<A, 0> or dynamic_dimension<B, 0>) detail::solve_check_A_and_B_rows_match(a, b);
        return make_self_contained(std::forward<B>(b) / constant_coefficient{a}());
      }
      else if constexpr (constant_matrix<B>)
      {
        if constexpr (dynamic_dimension<A, 0> or dynamic_dimension<B, 0>) detail::solve_check_A_and_B_rows_match(a, b);

        return make_constant_matrix_like<B>(
          constant_coefficient{b} / (internal::index_dimension_scalar_constant_of<1>(a) * constant_coefficient{a}),
          get_vector_space_descriptor<1>(a), get_vector_space_descriptor<1>(b));
      }
      else if constexpr (index_dimension_of_v<A, 0> == 1 or index_dimension_of_v<B, 0> == 1 or
        (not must_be_exact and (not must_be_unique or
          (not has_dynamic_dimensions<A> and index_dimension_of_v<A, 0> >= index_dimension_of_v<A, 1>))))
      {
        if constexpr (dynamic_dimension<A, 0> or dynamic_dimension<B, 0>) detail::solve_check_A_and_B_rows_match(a, b);
        return make_self_contained(b / (get_index_dimension_of<1>(a) * constant_coefficient_v<A>));
      }
      else //< The solution will be non-exact unless every row of b is identical.
      {
        return interface::library_interface<std::decay_t<A>>::template solve<must_be_unique, must_be_exact>(
          std::forward<A>(a), std::forward<B>(b));
      }
    }
    else if constexpr (diagonal_matrix<A> or
      ((index_dimension_of_v<A, 0> == 1 or index_dimension_of_v<B, 0> == 1) and index_dimension_of_v<A, 1> == 1))
    {
      auto op = [](auto&& b_elem, auto&& a_elem) {
        if (a_elem == 0)
        {
          if constexpr (not std::numeric_limits<scalar_type_of_t<B>>::has_infinity) throw std::logic_error {
            "In solve function, an element should be infinite, but the scalar type does not have infinite values"};
          else return std::numeric_limits<scalar_type_of_t<B>>::infinity();
        }
        else
        {
          return std::forward<decltype(b_elem)>(b_elem) / static_cast<scalar_type_of_t<B>>(std::forward<decltype(a_elem)>(a_elem));
        }
      };
      return n_ary_operation(get_all_dimensions_of(b), std::move(op), std::forward<B>(b), diagonal_of(std::forward<A>(a)));
    }
    else
    {
      return interface::library_interface<std::decay_t<A>>::template solve<must_be_unique, must_be_exact>(
        std::forward<A>(a), std::forward<B>(b));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_SOLVE_HPP
