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
 * \brief Overloaded general linear-algebra functions.
 */

#ifndef OPENKALMAN_LINEAR_ALGEBRA_FUNCTIONS_HPP
#define OPENKALMAN_LINEAR_ALGEBRA_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  /**
   * \brief Take the conjugate of a matrix
   * \tparam Arg The matrix
   */
  template<typename Arg>
  constexpr decltype(auto) conjugate(Arg&& arg) noexcept
  {
    if constexpr (not complex_number<scalar_type_of_t<Arg>> or zero_matrix<Arg> or identity_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (std::imag(constant_coefficient_v<Arg>) == 0)
        return std::forward<Arg>(arg);
      else
        return interface::LinearAlgebra<std::decay_t<Arg>>::conjugate(std::forward<Arg>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (std::imag(constant_diagonal_coefficient_v<Arg>) == 0)
        return std::forward<Arg>(arg);
      else
        return interface::LinearAlgebra<std::decay_t<Arg>>::conjugate(std::forward<Arg>(arg));
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<Arg>>::conjugate(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Take the transpose of a matrix
   * \tparam Arg The matrix
   */
  template<typename Arg>
  constexpr decltype(auto) transpose(Arg&& arg) noexcept
  {
    if constexpr (diagonal_matrix<Arg> or (self_adjoint_matrix<Arg> and not complex_number<scalar_type_of_t<Arg>>) or
      (constant_matrix<Arg> and square_matrix<Arg>))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<Arg>>::transpose(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Take the adjoint of a matrix
   * \tparam Arg The matrix
   */
  template<typename Arg>
  constexpr decltype(auto) adjoint(Arg&& arg) noexcept
  {
    if constexpr (self_adjoint_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg> or not complex_number<scalar_type_of_t<Arg>>)
    {
      return transpose(std::forward<Arg>(arg));
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (std::imag(constant_coefficient_v<Arg>) == 0)
        return transpose(std::forward<Arg>(arg));
      else if constexpr (not has_dynamic_dimensions<Arg> and row_dimension_of_v<Arg> == column_dimension_of_v<Arg>)
        return conjugate(std::forward<Arg>(arg));
      else
        return interface::LinearAlgebra<std::decay_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (std::imag(constant_diagonal_coefficient_v<Arg>) == 0)
        return transpose(std::forward<Arg>(arg));
      else if constexpr (not has_dynamic_dimensions<Arg> and row_dimension_of_v<Arg> == column_dimension_of_v<Arg>)
        return conjugate(std::forward<Arg>(arg));
      else
        return interface::LinearAlgebra<std::decay_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Take the determinant of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<typename Arg> requires has_dynamic_dimensions<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<has_dynamic_dimensions<Arg> or square_matrix<Arg>, int> = 0>
#endif
  constexpr auto determinant(Arg&& arg)
  {
    if constexpr (has_dynamic_dimensions<Arg>) if (get_dimensions_of<0>(arg) != get_dimensions_of<1>(arg))
      throw std::domain_error {
        "In determinant, rows of arg (" + std::to_string(get_index_dimension_of<0>(arg)) + ") do not match columns of arg (" +
        std::to_string(get_index_dimension_of<1>(arg)) + ")"};

    using Scalar = scalar_type_of_t<Arg>;

    if constexpr (identity_matrix<Arg>)
    {
      if constexpr (complex_number<Scalar>) return std::real(Scalar(1));
      else return Scalar(1);
    }
    else if constexpr (zero_matrix<Arg> or (constant_matrix<Arg> and not one_by_one_matrix<Arg>))
    {
      if constexpr (complex_number<Scalar>) return std::real(Scalar(0));
      else return Scalar(0);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return constant_coefficient_v<Arg>; //< One-by-one case. General case is handled above.
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (dynamic_rows<Arg>)
        return std::pow(constant_diagonal_coefficient_v<Arg>, get_index_dimension_of<0>(arg));
      else
        return internal::constexpr_pow(constant_diagonal_coefficient_v<Arg>, row_dimension_of_v<Arg>);
    }
    else if constexpr (one_by_one_matrix<Arg> and element_gettable<Arg, std::size_t, std::size_t>)
    {
      return get_element(arg, std::size_t(0), std::size_t(0));
    }
    else
    {
      auto r = interface::LinearAlgebra<std::decay_t<Arg>>::determinant(std::forward<Arg>(arg));
      static_assert(std::is_convertible_v<decltype(r), const scalar_type_of_t<Arg>>);
      if constexpr (self_adjoint_matrix<Arg> and complex_number<std::decay_t<decltype(r)>>) return std::real(r);
      else return r;
    }
  }


#ifdef __cpp_concepts
  /**
   * \brief Take the trace of a matrix
   * \tparam Arg The matrix
   */
  template<typename Arg> requires has_dynamic_dimensions<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<(has_dynamic_dimensions<Arg> or square_matrix<Arg>), int> = 0>
#endif
  constexpr auto trace(Arg&& arg)
  {
    if constexpr (has_dynamic_dimensions<Arg>) if (get_dimensions_of<0>(arg) != get_dimensions_of<1>(arg))
      throw std::domain_error {
        "In trace, rows of arg (" + std::to_string(get_index_dimension_of<0>(arg)) + ") do not match columns of arg (" +
        std::to_string(get_index_dimension_of<1>(arg)) + ")"};

    using Scalar = scalar_type_of_t<Arg>;

    if constexpr (identity_matrix<Arg>)
    {
      return Scalar(get_index_dimension_of<0>(arg));
    }
    else if constexpr (zero_matrix<Arg>)
    {
      return Scalar(0);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return Scalar(constant_coefficient_v<Arg> * get_index_dimension_of<0>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return Scalar(constant_diagonal_coefficient_v<Arg> * get_index_dimension_of<0>(arg));
    }
    else if constexpr (one_by_one_matrix<Arg> and element_gettable<Arg, std::size_t, std::size_t>)
    {
      return get_element(std::forward<Arg>(arg), std::size_t(0), std::size_t(0));
    }
    else
    {
      auto r = interface::LinearAlgebra<std::decay_t<Arg>>::trace(std::forward<Arg>(arg));
      static_assert(std::is_convertible_v<decltype(r), const scalar_type_of_t<Arg>>);
      return r;
    }
  }


  /**
   * \brief Do a rank update on a matrix, treating it as a self-adjoint matrix.
   * \details If A is not hermitian, the result will modify only the specified storage triangle. The contents of the
   * other elements outside the specified storage triangle are undefined.
   * - The update is A += αUU<sup>*</sup>, returning the updated hermitian A.
   * - If A is an lvalue reference and is writable, it will be updated in place and the return value will be an
   * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
   * \tparam t Whether to use the upper triangle elements (TriangleType::upper), lower triangle elements
   * (TriangleType::lower) or diagonal elements (TriangleType::diagonal).
   * \tparam A The matrix to be rank updated.
   * \tparam U The update vector or matrix.
   * \returns an updated native, writable matrix in hermitian form.
   */
#ifdef __cpp_concepts
  template<TriangleType t, typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (dynamic_rows<U> or dynamic_rows<A> or row_dimension_of_v<U> == row_dimension_of_v<A>) and
    (has_dynamic_dimensions<A> or square_matrix<A>)
#else
  template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (dynamic_rows<U> or dynamic_rows<A> or
      row_dimension_of<std::decay_t<U>>::value == row_dimension_of<std::decay_t<A>>::value) and
    (has_dynamic_dimensions<A> or square_matrix<A>), int> = 0>
#endif
  inline decltype(auto)
  rank_update_self_adjoint(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<0>(u))
      throw std::domain_error {
        "In rank_update_self_adjoint, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
        ") do not match rows of u (" + std::to_string(get_index_dimension_of<0>(u)) + ")"};

    if constexpr (has_dynamic_dimensions<A>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<1>(a))
      throw std::domain_error {
        "In rank_update_self_adjoint, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
        ") do not match columns of a (" + std::to_string(get_index_dimension_of<1>(a)) + ")"};

    if constexpr (zero_matrix<U>)
    {
      return std::forward<A>(a);
    }
    // \todo Generalize zero and diagonal cases from eigen and special-matrix
    else
    {
      using Trait = interface::LinearAlgebra<std::decay_t<A>>;
      return Trait::template rank_update_self_adjoint<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
  }


  /**
   * \overload
   * \brief The triangle type is derived from A, or is TriangleType::lower by default.
   */
#ifdef __cpp_concepts
  template<typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (dynamic_rows<U> or dynamic_rows<A> or row_dimension_of_v<U> == row_dimension_of_v<A>) and
    (has_dynamic_dimensions<A> or square_matrix<A>)
#else
  template<typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (dynamic_rows<U> or dynamic_rows<A> or row_dimension_of<U>::value == row_dimension_of<A>::value) and
    (has_dynamic_dimensions<A> or square_matrix<A>), int> = 0>
#endif
  inline decltype(auto)
  rank_update_self_adjoint(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (self_adjoint_matrix<A>)
      return rank_update_self_adjoint<self_adjoint_triangle_type_of_v<A>>(std::forward<A>(a), std::forward<U>(u), alpha);
    else if constexpr (triangular_matrix<A>)
      return rank_update_self_adjoint<triangle_type_of_v<A>>(std::forward<A>(a), std::forward<U>(u), alpha);
    else
      return rank_update_self_adjoint<TriangleType::lower>(std::forward<A>(a), std::forward<U>(u), alpha);
  }


  /**
   * \brief Do a rank update on a matrix, treating it as a triangular matrix.
   * \details If A is not a triangular matrix, the result will modify only the specified triangle. The contents of
   * other elements outside the specified triangle are undefined.
   * - If A is lower-triangular, diagonal, or one-by-one, the update is AA<sup>*</sup> += αUU<sup>*</sup>,
   * returning the updated A.
   * - If A is upper-triangular, the update is A<sup>*</sup>A += αUU<sup>*</sup>, returning the updated A.
   * - If A is an lvalue reference and is writable, it will be updated in place and the return value will be an
   * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
   * \tparam t Whether to use the upper triangle elements (TriangleType::upper), lower triangle elements
   * (TriangleType::lower) or diagonal elements (TriangleType::diagonal).
   * \tparam A The matrix to be rank updated.
   * \tparam U The update vector or matrix.
   * \returns an updated native, writable matrix in triangular (or diagonal) form.
   */
# ifdef __cpp_concepts
  template<TriangleType t, typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (t != TriangleType::lower or lower_triangular_matrix<A> or not upper_triangular_matrix<A>) and
    (t != TriangleType::upper or upper_triangular_matrix<A> or not lower_triangular_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<U> or row_dimension_of_v<A> == row_dimension_of_v<U>) and
    (has_dynamic_dimensions<A> or square_matrix<A>)
# else
  template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (t != TriangleType::lower or lower_triangular_matrix<A> or not upper_triangular_matrix<A>) and
    (t != TriangleType::upper or upper_triangular_matrix<A> or not lower_triangular_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<U> or row_dimension_of<A>::value == row_dimension_of<U>::value) and
    (has_dynamic_dimensions<A> or square_matrix<A>), int> = 0>
# endif
  inline decltype(auto)
  rank_update_triangular(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<0>(u))
      throw std::domain_error {
        "In rank_update_triangular, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
        ") do not match rows of u (" + std::to_string(get_index_dimension_of<0>(u)) + ")"};

    if constexpr (has_dynamic_dimensions<A>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<1>(a))
      throw std::domain_error {
        "In rank_update_triangular, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
        ") do not match columns of a (" + std::to_string(get_index_dimension_of<1>(a)) + ")"};

    if constexpr (zero_matrix<U>)
    {
      return std::forward<A>(a);
    }
    // \todo Generalize zero and diagonal cases from eigen and special-matrix
    else
    {
      using Trait = interface::LinearAlgebra<std::decay_t<A>>;
      return Trait::template rank_update_triangular<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
  }


  /**
   * \overload
   * \brief The triangle type is derived from A, or is TriangleType::lower by default.
   */
# ifdef __cpp_concepts
  template<typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (dynamic_rows<A> or dynamic_rows<U> or row_dimension_of_v<A> == row_dimension_of_v<U>) and
    (has_dynamic_dimensions<A> or square_matrix<A>)
# else
  template<typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (dynamic_rows<A> or dynamic_rows<U> or row_dimension_of<A>::value == row_dimension_of<U>::value) and
    (has_dynamic_dimensions<A> or square_matrix<A>), int> = 0>
# endif
  inline decltype(auto)
  rank_update_triangular(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (triangular_matrix<A>)
      return rank_update_triangular<triangle_type_of_v<A>>(std::forward<A>(a), std::forward<U>(u), alpha);
    else if constexpr (self_adjoint_matrix<A>)
      return rank_update_triangular<self_adjoint_triangle_type_of_v<A>>(std::forward<A>(a), std::forward<U>(u), alpha);
    else
      return rank_update_triangular<TriangleType::lower>(std::forward<A>(a), std::forward<U>(u), alpha);
  }


  /**
   * \brief Do a rank update on a hermitian, triangular, or one-by-one matrix.
   * \details
   * - If A is hermitian and non-diagonal, then the update is A += αUU<sup>*</sup>, returning the updated hermitian A.
   * - If A is lower-triangular, diagonal, or one-by-one, the update is AA<sup>*</sup> += αUU<sup>*</sup>,
   * returning the updated A.
   * - If A is upper-triangular, the update is A<sup>*</sup>A += αUU<sup>*</sup>, returning the updated A.
   * - If A is an lvalue reference and is writable, it will be updated in place and the return value will be an
   * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
   * \tparam A A matrix to be updated, which must be hermitian (known at compile time),
   * triangular (known at compile time), or one-by-one (determinable at runtime) .
   * \tparam U A matrix or column vector with a number of rows that matches the size of A.
   * \param alpha A scalar multiplication factor.
   */
#ifdef __cpp_concepts
  template<typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (triangular_matrix<A> or self_adjoint_matrix<A> or (zero_matrix<A> and has_dynamic_dimensions<A>) or
      (constant_matrix<A> and not complex_number<scalar_type_of<A>> and has_dynamic_dimensions<A>) or
      (dynamic_rows<A> and dynamic_columns<A>) or (dynamic_rows<A> and column_vector<A>) or
      (dynamic_columns<A> and row_vector<A>)) and
    (dynamic_rows<U> or dynamic_rows<A> or row_dimension_of_v<U> == row_dimension_of_v<A>) and
    (has_dynamic_dimensions<A> or square_matrix<A>)
#else
  template<typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (triangular_matrix<A> or self_adjoint_matrix<A> or (zero_matrix<A> and has_dynamic_dimensions<A>) or
      (constant_matrix<A> and not complex_number<scalar_type_of<A>> and has_dynamic_dimensions<A>) or
      (dynamic_rows<A> and dynamic_columns<A>) or (dynamic_rows<A> and column_vector<A>) or
      (dynamic_columns<A> and row_vector<A>)) and
    (dynamic_rows<U> or dynamic_rows<A> or row_dimension_of<U>::value == row_dimension_of<A>::value) and
    (has_dynamic_dimensions<A> or square_matrix<A>), int> = 0>
#endif
  inline decltype(auto)
  rank_update(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (get_dimensions_of<0>(a) != get_dimensions_of<0>(u))
      throw std::domain_error {
        "In rank_update, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) + ") do not match rows of u (" +
        std::to_string(get_index_dimension_of<0>(u)) + ")"};

    if constexpr (triangular_matrix<A>)
    {
      constexpr TriangleType t = triangle_type_of_v<A>;
      return rank_update_triangular<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else if constexpr (zero_matrix<A> and has_dynamic_dimensions<A>)
    {
      return rank_update_triangular<TriangleType::diagonal>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else if constexpr (self_adjoint_matrix<A>)
    {
      constexpr TriangleType t = self_adjoint_triangle_type_of_v<A>;
      return rank_update_self_adjoint<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else if constexpr (constant_matrix<A> and not complex_number<scalar_type_of<A>> and has_dynamic_dimensions<A>)
    {
      return rank_update_self_adjoint<TriangleType::lower>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else
    {
      if constexpr (has_dynamic_dimensions<A>)
        if ((dynamic_rows<A> and get_index_dimension_of<0>(a) != 1) or (dynamic_columns<A> and get_index_dimension_of<1>(a) != 1))
          throw std::domain_error {
            "Non hermitian, non-triangular argument to rank_update expected to be one-by-one, but instead it has " +
            std::to_string(get_index_dimension_of<0>(a)) + " rows and " + std::to_string(get_index_dimension_of<1>(a)) + " columns"};

      auto e = std::sqrt(trace(a) * trace(a.conjugate()) + alpha * trace(u * adjoint(u)));

      if constexpr (std::is_lvalue_reference_v<A> and not std::is_const_v<std::remove_reference_t<A>> and writable<A>)
      {
        if constexpr (element_settable<A, std::size_t, std::size_t>)
        {
          set_element(a, e, 0, 0);
        }
        else
        {
          a = MatrixTraits<A>::make(e);
        }
        return std::forward<A>(a);
      }
      else
      {
        return MatrixTraits<A>::make(e);
      }
    }
  }


  namespace detail
  {
    template<typename A, typename B>
    void solve_check_A_and_B_rows_match(const A& a, const B& b)
    {
      if (get_dimensions_of<0>(a) != get_dimensions_of<0>(b))
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
   * such as if the equation is over-determined. * If <code>false<code>, then the function will return an estimate
   * instead of throwing an exception.
   * \tparam A The matrix A in the equation AX = B
   * \tparam B The matrix B in the equation AX = B
   * \return The unique solution X of the equation AX = B. If <code>must_be_unique</code>, then the function can return
   * any valid solution for X. In particular, if <code>must_be_unique</code>, the function has the following behavior:
   * - If A is a \ref zero_matrix, then the result X will also be a \ref zero_matrix
   */
  #ifdef __cpp_concepts
  template<bool must_be_unique = false, bool must_be_exact = false, typename A, typename B> requires
    (dynamic_rows<A> or dynamic_rows<B> or row_dimension_of_v<A> == row_dimension_of_v<B>) and
    (not zero_matrix<A> or not zero_matrix<B> or not must_be_unique) and
    (not zero_matrix<A> or not (constant_matrix<B> or constant_diagonal_matrix<B>) or zero_matrix<B> or not must_be_exact) and
    (not constant_matrix<A> or not constant_diagonal_matrix<B> or has_dynamic_dimensions<A> or
      (row_dimension_of_v<A> <= column_dimension_of_v<A> and row_dimension_of_v<B> <= column_dimension_of_v<A>) or
      (row_dimension_of_v<A> == 1 and row_dimension_of_v<B> == 1) or not must_be_exact)
  #else
  template<bool must_be_unique = false, bool must_be_exact = false, typename A, typename B, std::enable_if_t<
    (dynamic_rows<A> or dynamic_rows<B> or row_dimension_of_v<A> == row_dimension_of_v<B>) and
    (not zero_matrix<A> or not zero_matrix<B> or not must_be_unique) and
    (not zero_matrix<A> or not (constant_matrix<B> or constant_diagonal_matrix<B>) or zero_matrix<B> or not must_be_exact) and
    (not constant_matrix<A> or not constant_diagonal_matrix<B> or has_dynamic_dimensions<A> or
      (row_dimension_of_v<A> <= column_dimension_of_v<A> and row_dimension_of_v<B> <= column_dimension_of_v<A>) or
      (row_dimension_of_v<A> == 1 and row_dimension_of_v<B> == 1) or not must_be_exact), int> = 0>
  #endif
  constexpr auto
  solve(A&& a, B&& b)
  {
    using Scalar = scalar_type_of_t<A>;

    if constexpr (zero_matrix<B>)
    {
      if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);

      if constexpr (must_be_unique and not constant_matrix<A> and not constant_diagonal_matrix<A>)
      {
        if (a == make_zero_matrix_like(a))
          throw std::runtime_error {"solve function requires a unique solution, "
            "but because operands A and B are both zero matrices, result X may take on any value"};
        else
          return make_zero_matrix_like<B, Scalar>(get_dimensions_of<1>(a), get_dimensions_of<1>(b));
      }
      else
        return make_zero_matrix_like<B, Scalar>(get_dimensions_of<1>(a), get_dimensions_of<1>(b));
    }
    else if constexpr (zero_matrix<A>) //< This will be a non-exact solution unless b is zero.
    {
      if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);
        return make_zero_matrix_like<B, Scalar>(get_dimensions_of<1>(a), get_dimensions_of<1>(b));
    }
    else if constexpr (constant_diagonal_matrix<A>)
    {
      if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);
      if constexpr (identity_matrix<A>)
        return std::forward<B>(b);
      else
        return make_self_contained(std::forward<B>(b) / constant_diagonal_coefficient_v<A>);
    }
    else if constexpr (constant_matrix<A>)
    {
      constexpr auto a_const = constant_coefficient_v<A>;

      if constexpr ((row_dimension_of_v<A> == 1 or row_dimension_of_v<B> == 1) and column_dimension_of_v<A> == 1)
      {
        if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);
        return make_self_contained(std::forward<B>(b) / a_const);
      }
      else if constexpr (constant_matrix<B>)
      {
        if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);

        constexpr auto b_const = constant_coefficient_v<B>;
        constexpr auto a_cols = column_dimension_of_v<A>;

        if constexpr (a_cols == dynamic_size)
        {
          auto a_runtime_cols = get_index_dimension_of<1>(a);
          auto c = static_cast<Scalar>(b_const) / (a_runtime_cols * a_const);
          return make_self_contained(c * make_constant_matrix_like<B, 1, Scalar>(Dimensions{a_runtime_cols}, get_dimensions_of<1>(b)));
        }
        else
        {
  #if __cpp_nontype_template_args >= 201911L
          constexpr auto c = static_cast<Scalar>(b_const) / (a_cols * a_const);
          return make_constant_matrix_like<B, c, Scalar>(Dimensions<a_cols>{}, get_dimensions_of<1>(b));
  #else
          if constexpr(b_const % (a_cols * a_const) == 0)
          {
            constexpr std::size_t c = static_cast<std::size_t>(b_const) / (a_cols * static_cast<std::size_t>(a_const));
            return make_constant_matrix_like<B, c, Scalar>(Dimensions<a_cols>{}, get_dimensions_of<1>(b));
          }
          else
          {
            auto c = static_cast<Scalar>(b_const) / (a_cols * a_const);
            auto m = make_constant_matrix_like<B, 1, Scalar>(Dimensions<a_cols>{}, get_dimensions_of<1>(b));
            return make_self_contained(c * to_native_matrix<B>(m));
          }
  #endif
        }
      }
      else if constexpr (row_dimension_of_v<A> == 1 or row_dimension_of_v<B> == 1 or
        (not must_be_exact and (not must_be_unique or
          (not has_dynamic_dimensions<A> and row_dimension_of_v<A> >= column_dimension_of_v<A>))))
      {
        if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);
        return make_self_contained(b / (get_index_dimension_of<1>(a) * constant_coefficient_v<A>));
      }
      else //< The solution will be non-exact unless every row of b is identical.
      {
        return interface::LinearAlgebra<std::decay_t<A>>::template solve<must_be_unique, must_be_exact>(
          std::forward<A>(a), std::forward<B>(b));
      }
    }
    else if constexpr (diagonal_matrix<A> or
      ((row_dimension_of_v<A> == 1 or row_dimension_of_v<B> == 1) and column_dimension_of_v<A> == 1))
    {
      auto op = [](auto&& b_elem, auto&& a_elem) {
        if (a_elem == 0)
        {
          using Scalar = scalar_type_of_t<B>;
          if constexpr (not std::numeric_limits<Scalar>::has_infinity) throw std::logic_error {
            "In solve function, an element should be infinite, but the scalar type does not have infinite values"};
          else return std::numeric_limits<Scalar>::infinity();
        }
        else
        {
          return std::forward<decltype(b_elem)>(b_elem) / std::forward<decltype(a_elem)>(a_elem);
        }
      };
      return n_ary_operation(
        get_all_dimensions_of(b), std::move(op), std::forward<B>(b), diagonal_of(std::forward<A>(a)));
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<A>>::template solve<must_be_unique, must_be_exact>(
        std::forward<A>(a), std::forward<B>(b));
    }
  }


  /**
   * \brief Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * \tparam A The matrix to be decomposed
   * \returns L as a \ref lower_triangular_matrix
   */
#ifdef __cpp_concepts
  template<indexible A> requires (not euclidean_transformed<A>)
#else
  template<typename A, std::enable_if_t<indexible<A> and (not euclidean_transformed<A>), int> = 0>
#endif
  constexpr auto
  LQ_decomposition(A&& a)
  {
    if constexpr (lower_triangular_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (zero_matrix<A>)
    {
      auto dim = get_dimensions_of<0>(a);
      return make_zero_matrix_like<A>(dim, dim);
    }
    else if constexpr (constant_matrix<A>)
    {
      using Scalar = scalar_type_of_t<A>;
      constexpr auto constant = constant_coefficient_v<A>;

      const Scalar elem = constant * [](A&& a){
        if constexpr (dynamic_dimension<A, 1>) return std::sqrt(static_cast<Scalar>(get_index_dimension_of<1>(a)));
        else return internal::constexpr_sqrt(static_cast<Scalar>(index_dimension_of_v<A, 1>));
      }(std::forward<A>(a));

      if constexpr (dynamic_dimension<A, 0>)
      {
        auto dim = Dimensions {get_index_dimension_of<0>(a)};
        auto col1 = elem * make_constant_matrix_like<A, 1>(dim, Dimensions<1>{});

        auto ret = make_default_dense_writable_matrix_like<A>(dim, dim);

        if (get_dimension_size_of(dim) == 1)
          ret = std::move(col1);
        else
          ret = concatenate_horizontal(std::move(col1), make_zero_matrix_like<A>(dim, dim - Dimensions<1>{}));

        return SquareRootCovariance {std::move(ret), get_dimensions_of<0>(a)};
      }
      else
      {
        auto m = [](Scalar elem){
          constexpr auto dim = index_dimension_of_v<A, 0>;
          auto col1 = elem * make_constant_matrix_like<A, 1>(Dimensions<dim>{}, Dimensions<1>{});
          if constexpr (dimension_size_of_v<dim> == 1) return col1;
          else return concatenate_horizontal(std::move(col1), make_zero_matrix_like<A>(Dimensions<dim>{}, Dimensions<dim - 1>{}));
        }(elem);

        auto ret = Eigen3::TriangularMatrix<decltype(m), TriangleType::lower> {std::move(m)};

        using C = coefficient_types_of_t<A, 0>;
        if constexpr (euclidean_index_descriptor<C>) return ret;
        else return SquareRootCovariance {std::move(ret), C{}};
      }
    }
    else
    {
      auto m = interface::LinearAlgebra<std::decay_t<A>>::LQ_decomposition(std::forward<A>(a));
      auto ret = Eigen3::TriangularMatrix<decltype(m), TriangleType::lower> {std::move(m)};

      // \todo remove the "and false" once SquareRootCovariance works:
      if constexpr (dynamic_dimension<A, 0> and false)
      {
        return SquareRootCovariance {std::move(ret), get_dimensions_of<0>(a)};
      }
      else
      {
        using C = coefficient_types_of_t<A, 0>;
        if constexpr (euclidean_index_descriptor<C>) return ret;
        else return SquareRootCovariance {std::move(ret), C{}};
      }
    }
  }


  /**
   * \brief Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * \tparam A The matrix to be decomposed
   * \returns U as an \ref upper_triangular_matrix
   */
#ifdef __cpp_concepts
  template<indexible A>
#else
  template<typename A, std::enable_if_t<indexible<A>, int> = 0>
#endif
  constexpr auto
  QR_decomposition(A&& a)
  {
    if constexpr (upper_triangular_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (zero_matrix<A>)
    {
      auto dim = get_dimensions_of<1>(a);
      return make_zero_matrix_like<A>(dim, dim);
    }
    else if constexpr (constant_matrix<A>)
    {
      using Scalar = scalar_type_of_t<A>;
      constexpr auto constant = constant_coefficient_v<A>;

      const Scalar elem = constant * [](A&& a){
        if constexpr (dynamic_dimension<A, 0>) return std::sqrt(static_cast<Scalar>(get_index_dimension_of<0>(a)));
        else return internal::constexpr_sqrt(static_cast<Scalar>(index_dimension_of_v<A, 0>));
      }(std::forward<A>(a));

      if constexpr (dynamic_dimension<A, 1>)
      {
        auto dim = Dimensions {get_index_dimension_of<1>(a)};
        auto row1 = elem * make_constant_matrix_like<A, 1>(Dimensions<1>{}, dim);

        auto ret = make_default_dense_writable_matrix_like<A>(dim, dim);

        if (get_dimension_size_of(dim) == 1)
          ret = std::move(row1);
        else
          ret = concatenate_vertical(std::move(row1), make_zero_matrix_like<A>(dim - Dimensions<1>{}, dim));

        return SquareRootCovariance {std::move(ret), get_dimensions_of<1>(a)};
      }
      else
      {
        auto m = [](Scalar elem){
          constexpr auto dim = index_dimension_of_v<A, 1>;
          auto row1 = elem * make_constant_matrix_like<A, 1>(Dimensions<1>{}, Dimensions<dim>{});
          if constexpr (dimension_size_of_v<dim> == 1) return row1;
          else return concatenate_vertical(std::move(row1), make_zero_matrix_like<A>(Dimensions<dim - 1>{}, Dimensions<dim>{}));
        }(elem);

        auto ret = Eigen3::TriangularMatrix<decltype(m), TriangleType::upper> {std::move(m)};

        using C = coefficient_types_of_t<A, 1>;
        if constexpr (euclidean_index_descriptor<C>) return ret;
        else return SquareRootCovariance {std::move(ret), C{}};
      }
    }
    else
    {
      auto m = interface::LinearAlgebra<std::decay_t<A>>::QR_decomposition(std::forward<A>(a));
      auto ret = Eigen3::TriangularMatrix<decltype(m), TriangleType::upper> {std::move(m)};

      // \todo remove the "and false" once SquareRootCovariance works:
      if constexpr (dynamic_dimension<A, 1> and false)
      {
        return SquareRootCovariance {std::move(ret), get_dimensions_of<1>(a)};
      }
      else
      {
        using C = coefficient_types_of_t<A, 1>;
        if constexpr (euclidean_index_descriptor<C>) return ret;
        else return SquareRootCovariance {std::move(ret), C{}};
      }
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_LINEAR_ALGEBRA_FUNCTIONS_HPP
