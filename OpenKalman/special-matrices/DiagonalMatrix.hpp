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
  template<eigen_matrix NestedMatrix> requires dynamic_columns<NestedMatrix> or column_vector<NestedMatrix>
#else
  template<typename NestedMatrix>
#endif
  struct DiagonalMatrix
    : OpenKalman::internal::MatrixBase<DiagonalMatrix<NestedMatrix>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(eigen_matrix<NestedMatrix>);
    static_assert(dynamic_columns<NestedMatrix> or column_vector<NestedMatrix>);
#endif

  private:

    using Base = OpenKalman::internal::MatrixBase<DiagonalMatrix, NestedMatrix>;

    static constexpr auto dimensions = MatrixTraits<NestedMatrix>::rows;

#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct is_constructible_from_diagonal : std::false_type {};

    template<typename T>
    struct is_constructible_from_diagonal<T, std::void_t<decltype(NestedMatrix {diagonal_of(std::declval<T>())})>>
      : std::true_type {};
#endif

  public:

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;


    /// Construct from a \ref diagonal_matrix, including eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, DiagonalMatrix>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions) and
      (dynamic_columns<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::columns == dimensions) and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (not std::is_base_of_v<DiagonalMatrix, std::decay_t<Arg>>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions) and
      (dynamic_columns<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::columns == dimensions) and
      std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))>, int> = 0>
#endif
    DiagonalMatrix(Arg&& arg) noexcept
      : Base {[](Arg&& arg) -> decltype(auto) {
        if constexpr (dynamic_rows<Arg> and not dynamic_rows<NestedMatrix>)
        {
          auto r = row_count(std::forward<Arg>(arg));
          if (r != dimensions) throw std::domain_error {"Dynamic size of diagonal argument (" + std::to_string(r) +
            ") does not match the fixed DiagonalMatrix size (" + std::to_string(dimensions) + ") in " + __func__ +
            " at line " + std::to_string(__LINE__) + " of " + __FILE__};
        }
        return diagonal_of(std::forward<Arg>(arg));
    }(std::forward<Arg>(arg))} {}


    /// Construct from a \ref column_vector.
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<typename Arg> requires (not diagonal_matrix<Arg>) and (column_vector<Arg> or dynamic_columns<Arg>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions) and
      requires(Arg&& arg) { NestedMatrix {std::forward<Arg>(arg)}; }
#else
    template<typename Arg, std::enable_if_t<
      (not diagonal_matrix<Arg>) and (column_vector<Arg> or dynamic_columns<Arg>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit DiagonalMatrix(Arg&& arg) noexcept
      : Base {[](Arg&& arg) -> decltype(auto) {
        if constexpr (dynamic_rows<Arg> and not dynamic_rows<NestedMatrix>)
        {
          auto r = row_count(std::forward<Arg>(arg));
          if (r != dimensions) throw std::domain_error {"DiagonalMatrix argument has " + std::to_string(r) +
            " rows, but should have " + std::to_string(dimensions) + " at line " + std::to_string(__LINE__) +
            " of " + __FILE__};
        }

        if constexpr (dynamic_columns<Arg>)
        {
          auto c = column_count(std::forward<Arg>(arg));
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
    requires (dynamic_shape<NestedMatrix> or sizeof...(Args) == dimensions) and
      requires(Args ... args) {
        NestedMatrix {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)};
      }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
        (dynamic_shape<NestedMatrix> or sizeof...(Args) == dimensions) and
        std::is_constructible_v<NestedMatrix, eigen_matrix_t<Scalar, sizeof...(Args), 1>>, int> = 0>
#endif
    DiagonalMatrix(Args ... args) : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}


    /**
     * \brief Construct from a list of scalar coefficients defining a square matrix.
     * \details Only the diagonal elements are extracted.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) == dimensions * dimensions) and
      (dimensions > 1) and requires(Args ... args) {
        NestedMatrix {diagonal_of(MatrixTraits<eigen_matrix_t<Scalar, dimensions, dimensions>>::make(
          static_cast<const Scalar>(args)...))};
      }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
        (sizeof...(Args) == dimensions * dimensions) and (dimensions > 1) and
        std::is_constructible_v<NestedMatrix, decltype(diagonal_of(
          MatrixTraits<eigen_matrix_t<Scalar, dimensions, dimensions>>::make(
            static_cast<const Scalar>(std::declval<Args>())...)))>, int> = 0>
#endif
    DiagonalMatrix(Args ... args) : Base {diagonal_of(
      MatrixTraits<eigen_matrix_t<Scalar, dimensions, dimensions>>::make(static_cast<const Scalar>(args)...))} {}


    /// Assign from another \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_diagonal_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, DiagonalMatrix>) and
      (MatrixTraits<Arg>::rows == dimensions) and modifiable<NestedMatrix, nested_matrix_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg> and
      (not std::is_base_of_v<DiagonalMatrix, std::decay_t<Arg>>) and
      (MatrixTraits<Arg>::rows == dimensions) and modifiable<NestedMatrix, nested_matrix_t<Arg>>, int> = 0>
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
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions) and
      (dynamic_shape<Arg> or square_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<zero_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      (not identity_matrix<NestedMatrix>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions) and
      (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not zero_matrix<NestedMatrix>)
      {
        if constexpr (dynamic_shape<Arg>)
          assert(row_count(arg) == column_count(arg));

        if constexpr (dynamic_rows<NestedMatrix>)
        {
          assert(row_count(this->nested_matrix()) == row_count(arg));
          this->nested_matrix() = ZeroMatrix<Scalar, dynamic_extent, 1> {row_count(arg)};
        }
        else
        {
          this->nested_matrix() = ZeroMatrix<Scalar, dimensions, 1> {};
        }
      }
      return *this;
    }


    /// Assign from an \ref identity_matrix, other than \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<identity_matrix Arg>
    requires (not eigen_diagonal_expr<Arg>) and (not zero_matrix<NestedMatrix>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions) and
      (dynamic_shape<Arg> or square_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<
      identity_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and (not zero_matrix<NestedMatrix>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions) and
      (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not identity_matrix<NestedMatrix>)
      {
        if constexpr (dynamic_shape<Arg>)
          assert(row_count(arg) == column_count(arg));

        if constexpr (dynamic_rows<NestedMatrix>)
        {
          assert(row_count(this->nested_matrix()) == row_count(arg));
          this->nested_matrix() = ConstantMatrix<Scalar, 1, dynamic_extent, 1> {row_count(arg)};
        }
        else
        {
          this->nested_matrix() = ConstantMatrix<Scalar, 1, dimensions, 1> {};
        }
      }
      return *this;
    }


    /// Assign from a general diagonal matrix, other than \ref eigen_diagonal_expr zero_matrix, or identity_matrix.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    requires (not eigen_diagonal_expr<Arg>) and (not zero_matrix<Arg>) and (not identity_matrix<Arg>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions) and
      modifiable<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (not eigen_diagonal_expr<Arg>) and (not zero_matrix<Arg>) and (not identity_matrix<Arg>) and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions) and
      modifiable<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (dynamic_rows<NestedMatrix>)
        assert(row_count(this->nested_matrix()) == row_count(arg));

      this->nested_matrix() = diagonal_of(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    requires (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions), int> = 0>
#endif
    auto& operator+=(Arg&& arg)
    {
      if constexpr (dynamic_rows<NestedMatrix>)
        assert(row_count(this->nested_matrix()) == row_count(arg));

      this->nested_matrix() += diagonal_of(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<diagonal_matrix Arg>
    requires (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (dynamic_rows<Arg> or dynamic_rows<NestedMatrix> or MatrixTraits<Arg>::rows == dimensions), int> = 0>
#endif
    auto& operator-=(Arg&& arg)
    {
      if constexpr (dynamic_rows<NestedMatrix>)
        assert(row_count(this->nested_matrix()) == row_count(arg));

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
    template<typename Arg> requires (MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<(MatrixTraits<Arg>::rows == dimensions), int> = 0>
#endif
    auto& operator*=(const DiagonalMatrix<Arg>& arg)
    {
      static_assert(MatrixTraits<Arg>::rows == dimensions);
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

#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
  template<diagonal_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<diagonal_matrix<Arg>, int> = 0>
#endif
  DiagonalMatrix(Arg&&) -> DiagonalMatrix<passable_t<decltype(diagonal_of(std::declval<Arg&&>()))>>;


#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
  template<typename Arg> requires (column_vector<Arg> or dynamic_columns<Arg>) and (not diagonal_matrix<Arg>)
#else
  template<typename Arg, std::enable_if_t<
    (column_vector<Arg> or dynamic_columns<Arg>) and not diagonal_matrix<Arg>, int> = 0>
#endif
  explicit DiagonalMatrix(Arg&&) -> DiagonalMatrix<passable_t<Arg>>;


#ifdef __cpp_concepts
  template<arithmetic_or_complex Arg, arithmetic_or_complex ... Args> requires (std::common_with<Arg, Args> and ...)
#else
  template<typename Arg, typename ... Args, std::enable_if_t<
    (arithmetic_or_complex<Arg> and ... and arithmetic_or_complex<Args>), int> = 0>
#endif
    DiagonalMatrix(const Arg&, const Args& ...) -> DiagonalMatrix<
      Eigen3::eigen_matrix_t<std::common_type_t<Arg, Args...>, 1 + sizeof...(Args), 1>>;

} // OpenKalman::Eigen3


namespace OpenKalman
{
  // --------------------------- //
  //        MatrixTraits         //
  // --------------------------- //

  namespace internal
  {
    template<typename ColumnVector>
    struct nested_matrix_type<Eigen3::DiagonalMatrix<ColumnVector>>
    {
      using type = ColumnVector;
    };
  }


  template<typename NestedMatrix>
  struct MatrixTraits<Eigen3::DiagonalMatrix<NestedMatrix>>
  {
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto rows = MatrixTraits<NestedMatrix>::rows;
    static constexpr auto columns = rows;

    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::DiagonalMatrix<NestedMatrix>>;

    template<std::size_t r = rows, std::size_t c = columns, typename S = Scalar>
    using NativeMatrixFrom = native_matrix_t<NestedMatrix, r, c, S>;

    using SelfContainedFrom = Eigen3::DiagonalMatrix<self_contained_t<NestedMatrix>>;

    template<TriangleType storage_triangle = TriangleType::diagonal, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<NativeMatrixFrom<dim, dim, S>, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::diagonal, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<NativeMatrixFrom<dim, dim, S>, triangle_type>;

    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<NativeMatrixFrom<dim, 1, S>>;


#ifdef __cpp_concepts
    template<column_vector Arg>
#else
    template<typename Arg, std::enable_if_t<column_vector<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      if constexpr (Eigen3::eigen_diagonal_expr<Arg>)
        return Eigen3::DiagonalMatrix<nested_matrix_t<Arg>> {std::forward<Arg>(arg)};
      else
        return Eigen3::DiagonalMatrix<Arg> {std::forward<Arg>(arg)};
    }


    /** Make diagonal matrix using a list of coefficients defining the diagonal.
     * The size of the list must match the number of diagonal coefficients.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args>
    requires (rows == dynamic_extent) or (sizeof...(Args) == rows)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      ((rows == dynamic_extent) or (sizeof...(Args) == rows)), int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...));
    }


    /** Make diagonal matrix using a list of coefficients in row-major order (ignoring non-diagonal coefficients).
     * The size of the list must match the number of coefficients in the matrix (diagonal and non-diagonal).
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args>
    requires (rows != dynamic_extent) and (rows > 1) and (sizeof...(Args) == rows * rows)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (rows != dynamic_extent) and (rows > 1) and (sizeof...(Args) == rows * rows), int> = 0>
#endif
    static auto
    make(const Args ... args)
    {
      return make(Eigen3::make_self_contained(Eigen3::diagonal_of(MatrixTraits<NativeMatrixFrom<>>::make(
        static_cast<const Scalar>(args)...))));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args>
    requires (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return MatrixTraits<NativeMatrixFrom<>>::zero(static_cast<std::size_t>(args)..., static_cast<std::size_t>(args)...);
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args>
    requires (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) == (rows == dynamic_extent ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      return MatrixTraits<NativeMatrixFrom<>>::identity(args...);
    }

  };

} // namespace OpenKalman


#endif //OPENKALMAN_EIGEN3_DIAGONALMATRIX_HPP