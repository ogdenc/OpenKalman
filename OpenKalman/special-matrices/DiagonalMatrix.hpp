/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for Eigen3::DiagonalMatrix
 */

#ifndef OPENKALMAN_EIGEN3_DIAGONALMATRIX_HPP
#define OPENKALMAN_EIGEN3_DIAGONALMATRIX_HPP

namespace OpenKalman::Eigen3
{

#ifdef __cpp_concepts
  template<typename NestedMatrix> requires dynamic_columns<NestedMatrix> or column_vector<NestedMatrix>
#else
  template<typename NestedMatrix>
#endif
  struct DiagonalMatrix
    : OpenKalman::internal::MatrixBase<DiagonalMatrix<NestedMatrix>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(dynamic_columns<NestedMatrix> or column_vector<NestedMatrix>);
#endif

  private:

    using Base = OpenKalman::internal::MatrixBase<DiagonalMatrix, NestedMatrix>;

    static constexpr auto dim = row_dimension_of_v<NestedMatrix>;

#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct is_constructible_from_diagonal : std::false_type {};

    template<typename T>
    struct is_constructible_from_diagonal<T, std::void_t<decltype(NestedMatrix {diagonal_of(std::declval<T>())})>>
      : std::true_type {};
#endif

  public:

    using Scalar = scalar_type_of_t<NestedMatrix>;


    /// Default constructor.
#ifdef __cpp_concepts
    DiagonalMatrix() requires std::default_initializable<NestedMatrix> and (not has_dynamic_dimensions<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<
      std::is_default_constructible_v<T> and (not has_dynamic_dimensions<NestedMatrix>), int> = 0>
    DiagonalMatrix()
#endif
      : Base {} {}


    /// Construct from a \ref diagonal_matrix, including eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<typename Arg> requires
      (diagonal_matrix<Arg> or (not column_vector<Arg> and not dynamic_columns<Arg>)) and
      (not std::derived_from<std::decay_t<Arg>, DiagonalMatrix>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of_v<Arg> == dim) and
      (dynamic_columns<Arg> or dynamic_rows<NestedMatrix> or column_dimension_of_v<Arg> == dim) and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<
      (diagonal_matrix<Arg> or (not column_vector<Arg> and not dynamic_columns<Arg>)) and
      (not std::is_base_of_v<DiagonalMatrix, std::decay_t<Arg>>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of_v<Arg> == dim) and
      (dynamic_columns<Arg> or dynamic_rows<NestedMatrix> or column_dimension_of<Arg>::value == dim) and
      std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))>, int> = 0>
#endif
    DiagonalMatrix(Arg&& arg)
      : Base {[](Arg&& arg) -> decltype(auto) {
        if constexpr (dynamic_rows<Arg> and not dynamic_rows<NestedMatrix>)
        {
          auto r = get_dimensions_of<0>(std::forward<Arg>(arg));
          if (r != dim) throw std::domain_error {"Dynamic size of diagonal argument (" + std::to_string(r) +
            ") does not match the fixed DiagonalMatrix size (" + std::to_string(dim) + ") in " + __func__ +
            " at line " + std::to_string(__LINE__) + " of " + __FILE__};
        }
        return diagonal_of(std::forward<Arg>(arg));
    }(std::forward<Arg>(arg))} {}


    /// Construct from a \ref column_vector.
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<typename Arg> requires (not diagonal_matrix<Arg>) and (column_vector<Arg> or dynamic_columns<Arg>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of_v<Arg> == dim) and
      requires(Arg&& arg) { NestedMatrix {std::forward<Arg>(arg)}; }
#else
    template<typename Arg, std::enable_if_t<
      (not diagonal_matrix<Arg>) and (column_vector<Arg> or dynamic_columns<Arg>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of<Arg>::value == dim) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit DiagonalMatrix(Arg&& arg)
      : Base {[](Arg&& arg) -> decltype(auto) {
        if constexpr (dynamic_rows<Arg> and not dynamic_rows<NestedMatrix>)
        {
          auto r = get_index_dimension_of<0>(std::forward<Arg>(arg));
          if (r != dim) throw std::domain_error {"DiagonalMatrix argument has " + std::to_string(r) +
            " rows, but should have " + std::to_string(dim) + " at line " + std::to_string(__LINE__) +
            " of " + __FILE__};
        }

        if constexpr (dynamic_columns<Arg>)
        {
          auto c = get_index_dimension_of<1>(std::forward<Arg>(arg));
          if (c != 1) throw std::domain_error {"DiagonalMatrix argument has " + std::to_string(c) +
            " columns, but should have 1 at line " + std::to_string(__LINE__) + " of " + __FILE__};
        }

        return std::forward<Arg>(arg);
    }(std::forward<Arg>(arg))} {}


    /**
     * \brief Construct from a list of scalar coefficients that define the diagonal.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args>
    requires (has_dynamic_dimensions<NestedMatrix> or sizeof...(Args) == dim) and
      requires(Args ... args) {
        NestedMatrix {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)};
      }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
        (has_dynamic_dimensions<NestedMatrix> or sizeof...(Args) == dim) and
        std::is_constructible_v<NestedMatrix, untyped_dense_writable_matrix_t<NestedMatrix, sizeof...(Args), 1>>, int> = 0>
#endif
    DiagonalMatrix(Args ... args) : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}


    /**
     * \brief Construct from a list of scalar coefficients defining a square matrix.
     * \details Only the diagonal elements are extracted.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) == dim * dim) and (dim > 1) and
      requires(Args ... args) {
        NestedMatrix {diagonal_of(make_dense_writable_matrix_from<NestedMatrix, dim, dim, Scalar>(
          static_cast<const Scalar>(args)...))};
      }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
        (sizeof...(Args) == dim * dim) and (dim > 1) and
        std::is_constructible_v<NestedMatrix, decltype(diagonal_of(
          make_dense_writable_matrix_from<NestedMatrix, dim, dim, Scalar>(
            static_cast<const Scalar>(std::declval<Args>())...)))>, int> = 0>
#endif
    DiagonalMatrix(Args ... args) : Base {diagonal_of(
      make_dense_writable_matrix_from<NestedMatrix, dim, dim, Scalar>(static_cast<const Scalar>(args)...))} {}


    /// Assign from another \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_diagonal_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, DiagonalMatrix>) and
      (row_dimension_of_v<Arg> == dim) and modifiable<NestedMatrix, nested_matrix_of_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg> and
      (not std::is_base_of_v<DiagonalMatrix, std::decay_t<Arg>>) and
      (row_dimension_of<Arg>::value == dim) and modifiable<NestedMatrix, nested_matrix_of_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = nested_matrix(std::forward<Arg>(other));
      }
      return *this;
    }


    /// Assign from a \ref zero_matrix, other than \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<zero_matrix Arg>
    requires (not eigen_diagonal_expr<Arg>) and (not identity_matrix<NestedMatrix>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of_v<Arg> == dim) and
      (has_dynamic_dimensions<Arg> or square_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<zero_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      (not identity_matrix<NestedMatrix>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of<Arg>::value == dim) and
      (has_dynamic_dimensions<Arg> or square_matrix<Arg>), int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not zero_matrix<NestedMatrix>)
      {
        if constexpr (has_dynamic_dimensions<Arg>) assert(get_dimensions_of<0>(arg) == get_dimensions_of<1>(arg));
        if constexpr (dynamic_rows<NestedMatrix>) assert(get_dimensions_of<0>(this->nested_matrix()) == get_dimensions_of<0>(arg));

        this->nested_matrix() = make_zero_matrix_like(this->nested_matrix());
      }
      return *this;
    }


    /// Assign from an \ref identity_matrix, other than \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<identity_matrix Arg>
    requires (not eigen_diagonal_expr<Arg>) and (not zero_matrix<NestedMatrix>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of_v<Arg> == dim) and
      (has_dynamic_dimensions<Arg> or square_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<
      identity_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and (not zero_matrix<NestedMatrix>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of<Arg>::value == dim) and
      (has_dynamic_dimensions<Arg> or square_matrix<Arg>), int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not identity_matrix<NestedMatrix>)
      {
        if constexpr (has_dynamic_dimensions<Arg>) assert(get_dimensions_of<0>(arg) == get_dimensions_of<1>(arg));
        if constexpr (dynamic_rows<NestedMatrix>) assert(get_dimensions_of<0>(this->nested_matrix()) == get_dimensions_of<0>(arg));

        this->nested_matrix() = make_constant_matrix_like<NestedMatrix, 1>(Dimensions<dim>{}, Dimensions<1>{});
      }
      return *this;
    }


    /// Assign from a general diagonal matrix, other than \ref eigen_diagonal_expr zero_matrix, or identity_matrix.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    requires (not eigen_diagonal_expr<Arg>) and (not zero_matrix<Arg>) and (not identity_matrix<Arg>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of_v<Arg> == dim) and
      modifiable<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (not eigen_diagonal_expr<Arg>) and (not zero_matrix<Arg>) and (not identity_matrix<Arg>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of<Arg>::value == dim) and
      modifiable<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (dynamic_rows<NestedMatrix>)
        assert(get_dimensions_of<0>(this->nested_matrix()) == get_dimensions_of<0>(arg));

      this->nested_matrix() = diagonal_of(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    requires (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of_v<Arg> == dim)
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of<Arg>::value == dim), int> = 0>
#endif
    auto& operator+=(Arg&& arg)
    {
      if constexpr (dynamic_rows<NestedMatrix>)
        assert(get_dimensions_of<0>(this->nested_matrix()) == get_dimensions_of<0>(arg));

      this->nested_matrix() += diagonal_of(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    requires (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of_v<Arg> == dim)
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or row_dimension_of<Arg>::value == dim), int> = 0>
#endif
    auto& operator-=(Arg&& arg)
    {
      if constexpr (dynamic_rows<NestedMatrix>)
        assert(get_dimensions_of<0>(this->nested_matrix()) == get_dimensions_of<0>(arg));

      this->nested_matrix() -= diagonal_of(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      this->nested_matrix() *= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      this->nested_matrix() /= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (row_dimension_of_v<Arg> == dim)
#else
    template<typename Arg, std::enable_if_t<(row_dimension_of<Arg>::value == dim), int> = 0>
#endif
    auto& operator*=(const DiagonalMatrix<Arg>& arg)
    {
      static_assert(row_dimension_of_v<Arg> == dim);
      this->nested_matrix() = this->nested_matrix().array() * arg.nested_matrix().array();
      return *this;
    }


    auto square() const
    {
      auto b = this->nested_matrix().array().square().matrix();
      return DiagonalMatrix<decltype(b)>(std::move(b));
    }


    auto square_root() const
    {
      auto b = this->nested_matrix().cwiseSqrt();
      return DiagonalMatrix<decltype(b)>(std::move(b));
    }

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

#if not defined(__cpp_concepts) or not OPENKALMAN_CPP_FEATURE_CONCEPTS
  namespace detail
  {
    template<typename T, typename = void>
    struct diagonal_exists : std::false_type {};

    template<typename T>
    struct diagonal_exists<T, std::void_t<decltype(diagonal_of(std::declval<T>()))>> : std::true_type {};
  };
#endif


  /**
   * \brief Deduce DiagonalMatrix template parameters for a diagonal matrix.
   * \tparam Arg A diagonal matrix
   */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
  template<typename Arg> requires
    (diagonal_matrix<Arg> or (not column_vector<Arg> and not dynamic_columns<Arg>)) and
    requires(Arg&& arg) { diagonal_of(std::forward<Arg>(arg)); }
#else
  template<typename Arg, std::enable_if_t<
    (diagonal_matrix<Arg> or (not column_vector<Arg> and not dynamic_columns<Arg>)) and
    detail::diagonal_exists<Arg>::value, int> = 0>
#endif
  DiagonalMatrix(Arg&&) -> DiagonalMatrix<std::conditional_t<std::is_lvalue_reference_v<Arg>,
    decltype(diagonal_of(std::declval<Arg&&>())),
    equivalent_self_contained_t<decltype(diagonal_of(std::declval<Arg&&>()))>>>;


  /**
   * \brief Deduce DiagonalMatrix template parameters for a column vector.
   * \tparam Arg A column vector or a matrix with dynamic columns that could be a column vector
   */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
  template<typename Arg> requires ((not diagonal_matrix<Arg>) and (column_vector<Arg> or dynamic_columns<Arg>)) or
    (not requires(Arg&& arg) { diagonal_of(std::forward<Arg>(arg)); })
#else
  template<typename Arg, std::enable_if_t<(not diagonal_matrix<Arg> and (column_vector<Arg> or dynamic_columns<Arg>)) or
    not detail::diagonal_exists<Arg>::value, int> = 0>
#endif
  explicit DiagonalMatrix(Arg&&) -> DiagonalMatrix<passable_t<Arg>>;


} // OpenKalman::Eigen3



#endif //OPENKALMAN_EIGEN3_DIAGONALMATRIX_HPP
