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
  template<column_vector NestedMatrix> requires eigen_matrix<NestedMatrix>
#else
  template<typename NestedMatrix>
#endif
  struct DiagonalMatrix
    : OpenKalman::internal::MatrixBase<DiagonalMatrix<NestedMatrix>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(column_vector<NestedMatrix>);
    static_assert(eigen_matrix<NestedMatrix>);
#endif

  private:

    using Base = OpenKalman::internal::MatrixBase<DiagonalMatrix, NestedMatrix>;

    static constexpr auto dimensions = MatrixTraits<NestedMatrix>::rows;

  public:

    using typename Base::Scalar;


    /// Default constructor.
#ifdef __cpp_concepts
    DiagonalMatrix() requires std::default_initializable<Base>
#else
    template<typename T = Base, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    DiagonalMatrix()
#endif
      : Base {} {}


    /// Copy constructor.
    DiagonalMatrix(const DiagonalMatrix& other) : Base {other} {}


    /// Move constructor.
    DiagonalMatrix(DiagonalMatrix&& other) noexcept: Base {std::move(other)} {}


    /// Construct from a compatible \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_diagonal_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, DiagonalMatrix>) and
      requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg> and
      (not std::is_base_of_v<DiagonalMatrix, std::decay_t<Arg>>) and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    DiagonalMatrix(Arg&& other) noexcept : Base {nested_matrix(std::forward<Arg>(other))} {}


    /// Construct from a zero matrix.
#if defined(__cpp_concepts) and defined(__cpp_conditional_explicit)
    template<zero_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and (column_vector<Arg> or square_matrix<Arg>) and
      std::constructible_from<NestedMatrix, ZeroMatrix<Scalar, dimensions, 1>&&>
    explicit (column_vector<Arg> and not square_matrix<Arg>)
    DiagonalMatrix(Arg&&) : Base {ZeroMatrix<Scalar, dimensions, 1> {}} {}
#else
    template<typename Arg, std::enable_if_t<zero_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      column_vector<Arg> and (not square_matrix<Arg>) and
      std::is_constructible_v<NestedMatrix, ZeroMatrix<Scalar, dimensions, 1>&&>, int> = 0>
    explicit DiagonalMatrix(Arg&&) : Base {ZeroMatrix<Scalar, dimensions, 1> {}} {}

    /// \overload
    template<typename Arg, std::enable_if_t<zero_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      (not column_vector<Arg> or square_matrix<Arg>) and
      std::is_constructible_v<NestedMatrix, ZeroMatrix<Scalar, dimensions, 1>&&>, int> = 0>
    DiagonalMatrix(Arg&&) : Base {ZeroMatrix<Scalar, dimensions, 1> {}} {}
#endif


    /// Construct from an identity matrix.
#ifdef __cpp_concepts
    template<identity_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and (not one_by_one_matrix<Arg>) and
      requires { std::constructible_from<NestedMatrix, ConstantMatrix<Scalar, 1, dimensions, 1>&&>; }
#else
    template<typename Arg, std::enable_if_t<identity_matrix<Arg> and
      (not eigen_diagonal_expr<Arg>) and (not one_by_one_matrix<Arg>) and
      std::is_constructible_v<NestedMatrix, ConstantMatrix<Scalar, 1, dimensions, 1>&&>, int> = 0>
#endif
    DiagonalMatrix(const Arg&) : Base {ConstantMatrix<Scalar, 1, dimensions, 1> {}} {}


    /// Construct from a compatible \ref diagonal_matrix, general case.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and
      (not zero_matrix<Arg>) and (not identity_matrix<Arg> or one_by_one_matrix<Arg>) and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      (not zero_matrix<Arg>) and (not identity_matrix<Arg> or one_by_one_matrix<Arg>) and
      std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))>, int> = 0>
#endif
    DiagonalMatrix(Arg&& arg) noexcept: Base {diagonal_of(std::forward<Arg>(arg))} {}


    /// Construct from a \ref column_vector \ref eigen_matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (not diagonal_matrix<Arg>) and column_vector<Arg> and
      (not zero_matrix<Arg>) and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (not diagonal_matrix<Arg>) and
      column_vector<Arg> and (not zero_matrix<Arg>) and std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit DiagonalMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a \ref square_matrix "square" \ref eigen_matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (not diagonal_matrix<Arg>) and (not column_vector<Arg>) and
      square_matrix<Arg> and requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and
      (not diagonal_matrix<Arg>) and (not column_vector<Arg>) and square_matrix<Arg> and
      std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>, int> = 0>
#endif
    explicit DiagonalMatrix(Arg&& arg) noexcept : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /**
     * \brief Construct from a list of scalar coefficients that define the diagonal.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) == dimensions) and
      requires(Args ... args) {
        NestedMatrix {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)};
      }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
        sizeof...(Args) == dimensions and
        std::is_constructible_v<NestedMatrix, eigen_matrix_t<Scalar, sizeof...(Args), 1>>, int> = 0>
#endif
    DiagonalMatrix(Args ... args) : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}


    /**
     * \brief Construct from a list of scalar coefficients defining a square matrix.
     * \details Only the diagonal elements are extracted.
     */
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
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


    /// Copy assignment operator.
    auto& operator=(const DiagonalMatrix& other)
    {
      Base::operator=(other);
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(DiagonalMatrix&& other) noexcept
    {
      Base::operator=(std::move(other));
      return *this;
    }


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
    template<zero_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and
      (not identity_matrix<NestedMatrix>) and (MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<zero_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      (not identity_matrix<NestedMatrix>) and (MatrixTraits<Arg>::rows == dimensions), int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero_matrix<NestedMatrix>)
      {
        this->nested_matrix() = ZeroMatrix<Scalar, dimensions, 1>();
      }
      return *this;
    }


    /// Assign from an \ref identity_matrix, other than \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<identity_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and
      (not zero_matrix<NestedMatrix>) and (MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<identity_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      (not zero_matrix<NestedMatrix>) and (MatrixTraits<Arg>::rows == dimensions), int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = ConstantMatrix<Scalar, 1, dimensions, 1> {};
      }
      return *this;
    }


    /// Assign from a general diagonal matrix, other than \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and
      (not zero_matrix<Arg>) and (not identity_matrix<Arg>) and (MatrixTraits<Arg>::rows == dimensions) and
      modifiable<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and (not eigen_diagonal_expr<Arg>) and
      (not zero_matrix<Arg>) and (not identity_matrix<Arg>) and (MatrixTraits<Arg>::rows == dimensions) and
      modifiable<NestedMatrix, decltype(diagonal_of(std::declval<Arg>()))>, int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = diagonal_of(std::forward<Arg>(other));
      }
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<(MatrixTraits<Arg>::rows == dimensions), int> = 0>
#endif
    auto& operator+=(const DiagonalMatrix<Arg>& arg)
    {
      this->nested_matrix() += arg.nested_matrix();
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<(MatrixTraits<Arg>::rows == dimensions), int> = 0>
#endif
    auto& operator-=(const DiagonalMatrix<Arg>& arg)
    {
      this->nested_matrix() -= arg.nested_matrix();
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


    auto operator()(std::size_t i, std::size_t j)
    {
      if constexpr (element_settable<DiagonalMatrix, 2>)
        return OpenKalman::internal::ElementAccessor(*this, i, j);
      else
        return std::as_const(*this)(i, j);
    }


    auto operator()(std::size_t i, std::size_t j) const noexcept
    {
      return OpenKalman::internal::ElementAccessor(*this, i, j);
    }


    auto operator[](std::size_t i)
    {
      if constexpr (element_settable<DiagonalMatrix, 1>)
        return OpenKalman::internal::ElementAccessor(*this, i);
      else
        return std::as_const(*this)[i];
    }


    auto operator[](std::size_t i) const { return OpenKalman::internal::ElementAccessor(*this, i); }


    auto operator()(std::size_t i) { return operator[](i); }


    auto operator()(std::size_t i) const { return operator[](i); }

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<diagonal_matrix Arg> requires (not eigen_diagonal_expr<Arg>) and (not column_vector<Arg>)
#else
  template<typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
    (not eigen_diagonal_expr<Arg>) and (not column_vector<Arg>), int> = 0>
#endif
  DiagonalMatrix(Arg&&) -> DiagonalMatrix<self_contained_t<decltype(diagonal_of(std::declval<Arg>()))>>;


#ifdef __cpp_concepts
  template<eigen_matrix Arg> requires square_matrix<Arg> and (not diagonal_matrix<Arg>) and (not column_vector<Arg>)
#else
  template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    square_matrix<Arg> and (not diagonal_matrix<Arg>) and (not column_vector<Arg>), int> = 0>
#endif
  explicit DiagonalMatrix(Arg&&) -> DiagonalMatrix<self_contained_t<decltype(diagonal_of(std::declval<Arg>()))>>;


  // Unlike SFINAE version, the concepts version incorrectly matches M==double in both GCC 10.1.0 and clang 10.0.0:
#if defined(__cpp_concepts) and false
  template<eigen_matrix M> requires column_vector<M>
#else
  template<typename M, std::enable_if_t<eigen_matrix<M> and column_vector<M>, int> = 0>
#endif
  explicit DiagonalMatrix(M&&) -> DiagonalMatrix<passable_t<M>>;


#ifdef __cpp_concepts
  template<typename Arg, typename ... Args> requires
    (std::is_arithmetic_v<std::decay_t<Arg>> and ... and std::is_arithmetic_v<std::decay_t<Args>>) and
    (std::common_with<Arg, Args> and ...)
#else
  template<typename Arg, typename ... Args, std::enable_if_t<
    (std::is_arithmetic_v<std::decay_t<Arg>> and ... and std::is_arithmetic_v<std::decay_t<Args>>), int> = 0>
#endif
    DiagonalMatrix(Arg&&, Args&& ...) -> DiagonalMatrix<
      Eigen::Matrix<std::common_type_t<std::decay_t<Arg>, std::decay_t<Args>...>, 1 + sizeof...(Args), 1>>;

} // OpenKalman::Eigen3


namespace OpenKalman
{
  // --------------------------- //
  //        MatrixTraits         //
  // --------------------------- //

  template<typename ArgType>
  struct MatrixTraits<Eigen3::DiagonalMatrix<ArgType>>
  {
    using NestedMatrix = ArgType;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto rows = MatrixTraits<NestedMatrix>::rows;
    static constexpr auto columns = rows;

    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::DiagonalMatrix<ArgType>>;

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
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) == rows)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (sizeof...(Args) == rows), int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...));
    }


    /** Make diagonal matrix using a list of coefficients in row-major order (ignoring non-diagonal coefficients).
     * The size of the list must match the number of coefficients in the matrix (diagonal and non-diagonal).
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires
      (sizeof...(Args) == rows * rows) and (rows > 1)
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...> and
      (sizeof...(Args) == rows * rows) and (rows > 1), int> = 0>
#endif
    static auto
    make(const Args ... args)
    {
      return make(Eigen3::make_self_contained(Eigen3::diagonal_of(MatrixTraits<NativeMatrixFrom<>>::make(
        static_cast<const Scalar>(args)...))));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return MatrixTraits<NativeMatrixFrom<>>::zero(static_cast<std::size_t>(args)...);
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      return MatrixTraits<NativeMatrixFrom<>>::identity(args...);
    }

  };

} // namespace OpenKalman


#endif //OPENKALMAN_EIGEN3_DIAGONALMATRIX_HPP
