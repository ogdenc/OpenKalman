/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief SelfAdjointMatrix and related definitions.
 */

#ifndef OPENKALMAN_EIGEN3_SELFADJOINTMATRIX_HPP
#define OPENKALMAN_EIGEN3_SELFADJOINTMATRIX_HPP

namespace OpenKalman::Eigen3
{
  using namespace OpenKalman::internal;

#ifdef __cpp_concepts
  template<typename NestedMatrix, TriangleType storage_triangle>
  requires (eigen_diagonal_expr<NestedMatrix> and not complex_number<typename MatrixTraits<NestedMatrix>::Scalar>) or
    (eigen_matrix<NestedMatrix> and (dynamic_shape<NestedMatrix> or square_matrix<NestedMatrix>))
#else
  template<typename NestedMatrix, TriangleType storage_triangle>
#endif
  struct SelfAdjointMatrix
    : OpenKalman::internal::MatrixBase<SelfAdjointMatrix<NestedMatrix, storage_triangle>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(eigen_diagonal_expr<NestedMatrix> or eigen_matrix<NestedMatrix>);
    static_assert(dynamic_shape<NestedMatrix> or square_matrix<NestedMatrix>);
#endif


  private:

    using Base = OpenKalman::internal::MatrixBase<SelfAdjointMatrix, NestedMatrix>;

    static constexpr auto uplo = storage_triangle == TriangleType::upper ? Eigen::Upper : Eigen::Lower;

    static constexpr auto dimensions =
      dynamic_rows<NestedMatrix> ? MatrixTraits<NestedMatrix>::columns : MatrixTraits<NestedMatrix>::rows;

  public:

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;


    /// Default constructor.
#ifdef __cpp_concepts
    SelfAdjointMatrix() requires std::default_initializable<NestedMatrix> and (not dynamic_shape<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<
      std::is_default_constructible_v<T> and (not dynamic_shape<NestedMatrix>), int> = 0>
    SelfAdjointMatrix()
#endif
      : Base {} {}


    /// Copy constructor.
    SelfAdjointMatrix(const SelfAdjointMatrix& other) : Base {other} {}


    /// Move constructor.
    SelfAdjointMatrix(SelfAdjointMatrix&& other) noexcept : Base {std::move(other)} {}


    /// Construct from a compatible self-adjoint matrix object of the same storage type
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      ((eigen_self_adjoint_expr<Arg> and self_adjoint_triangle_type_of_v<Arg> == self_adjoint_triangle_type_of_v<SelfAdjointMatrix>) or
       (eigen_triangular_expr<Arg> and diagonal_matrix<Arg>)) and
      (not eigen_diagonal_expr<NestedMatrix> or diagonal_matrix<nested_matrix_t<Arg>>) and
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
      //alt: requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } -- not accepted in GCC 10
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and
      ((eigen_self_adjoint_expr<Arg> and self_adjoint_triangle_type_of_v<Arg> == self_adjoint_triangle_type_of_v<SelfAdjointMatrix>) or
       (eigen_triangular_expr<Arg> and diagonal_matrix<Arg>)) and
      (not eigen_diagonal_expr<NestedMatrix> or diagonal_matrix<nested_matrix_t<Arg>>) and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from a compatible self-adjoint matrix object if NestedMatrix is \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      ((eigen_self_adjoint_expr<Arg> and self_adjoint_triangle_type_of_v<Arg> == self_adjoint_triangle_type_of_v<SelfAdjointMatrix>) or
       (eigen_triangular_expr<Arg> and diagonal_matrix<Arg>)) and
      eigen_diagonal_expr<NestedMatrix> and (not diagonal_matrix<nested_matrix_t<Arg>>) and
      requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and
      ((eigen_self_adjoint_expr<Arg> and self_adjoint_triangle_type_of_v<Arg> == self_adjoint_triangle_type_of_v<SelfAdjointMatrix>) or
       (eigen_triangular_expr<Arg> and diagonal_matrix<Arg>)) and
      eigen_diagonal_expr<NestedMatrix> and (not diagonal_matrix<nested_matrix_t<Arg>>) and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {to_diagonal(diagonal_of(nested_matrix(std::forward<Arg>(arg))))} {}


    /// Construct from a compatible self-adjoint matrix object of the opposite storage type
#ifdef __cpp_concepts
    template<eigen_self_adjoint_expr Arg>
    requires (not self_adjoint_triangle_type_of_v<Arg> == self_adjoint_triangle_type_of_v<SelfAdjointMatrix>) and
      requires(Arg&& arg) { NestedMatrix {transpose(nested_matrix(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> and
      (not self_adjoint_triangle_type_of_v<Arg> == self_adjoint_triangle_type_of_v<SelfAdjointMatrix>) and
      std::is_constructible_v<NestedMatrix, decltype(transpose(nested_matrix(std::declval<Arg&&>())))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {transpose(nested_matrix(std::forward<Arg>(arg)))} {}


    /// Construct from an \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_diagonal_expr Arg> requires std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<
      eigen_diagonal_expr<Arg> and std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a \ref self_adjoint_matrix "self-adjoint" \ref eigen_matrix
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires self_adjoint_matrix<Arg> and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and self_adjoint_matrix<Arg> and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a non-self-adjoint \ref eigen_matrix if NestedMatrix is not \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (not self_adjoint_matrix<Arg>) and (not eigen_diagonal_expr<NestedMatrix>) and
      (dynamic_shape<Arg> or square_matrix<Arg>) and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (not self_adjoint_matrix<Arg>) and
      (not eigen_diagonal_expr<NestedMatrix>) and (dynamic_shape<Arg> or square_matrix<Arg>) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit SelfAdjointMatrix(Arg&& arg) noexcept
      : Base {std::forward<Arg>((
        assert(not dynamic_rows<Arg> or dimensions == 0 or row_count(arg) == dimensions),
        arg))} {}


    /// Construct from an Eigen::SelfAdjointView of the same storage type.
#ifdef __cpp_concepts
    template<Eigen3::eigen_SelfAdjointView Arg>
    requires (self_adjoint_triangle_type_of_v<Arg> == self_adjoint_triangle_type_of_v<SelfAdjointMatrix>) and
      requires(Arg&& arg) { NestedMatrix {arg.nestedExpression()}; }
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_SelfAdjointView<Arg> and
      (self_adjoint_triangle_type_of_v<Arg> == self_adjoint_triangle_type_of_v<SelfAdjointMatrix>) and
      std::is_constructible_v<NestedMatrix, decltype(std::declval<Arg&&>().nestedExpression())>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) : Base {arg.nestedExpression()} {}


    /// Construct from a self-adjoint Eigen::SelfAdjointView of the opposite storage type.
#ifdef __cpp_concepts
    template<Eigen3::eigen_SelfAdjointView Arg>
    requires (not self_adjoint_triangle_type_of_v<Arg> == self_adjoint_triangle_type_of_v<SelfAdjointMatrix>) and
      requires(Arg&& arg) { NestedMatrix {transpose(arg.nestedExpression())}; }
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_SelfAdjointView<Arg> and
      (not self_adjoint_triangle_type_of_v<Arg> == self_adjoint_triangle_type_of_v<SelfAdjointMatrix>) and
      std::is_constructible_v<NestedMatrix, decltype(transpose(std::declval<Arg&&>().nestedExpression()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) : Base {transpose(arg.nestedExpression())} {}


    /// Construct from a non-self-adjoint \ref eigen_matrix if NestedMatrix is \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_matrix Arg>
    requires (not self_adjoint_matrix<Arg>) and eigen_diagonal_expr<NestedMatrix> and
      (dynamic_shape<Arg> or square_matrix<Arg>) and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t< eigen_matrix<Arg> and (not self_adjoint_matrix<Arg>) and
      eigen_diagonal_expr<NestedMatrix> and (dynamic_shape<Arg> or square_matrix<Arg>) and
      std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>, int> = 0>
#endif
    explicit SelfAdjointMatrix(Arg&& arg) noexcept : Base {DiagonalMatrix {diagonal_of(std::forward<Arg>(
      (dynamic_shape<Arg> ? (assert(row_count(arg) == column_count(arg)), arg) : arg)))}} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a self-adjoint matrix.
     * \note Operative if storage_triangle is not TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args>
    requires (sizeof...(Args) > 0) and (storage_triangle != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      requires(Args ... args) { NestedMatrix {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and
      (storage_triangle != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      (std::is_constructible_v<NestedMatrix,
          eigen_matrix_t<Scalar, constexpr_sqrt(sizeof...(Args)), constexpr_sqrt(sizeof...(Args))>> or
        (diagonal_matrix<NestedMatrix> and std::is_constructible_v<NestedMatrix,
          eigen_matrix_t<Scalar, sizeof...(Args), 1>>)), int> = 0>
#endif
    SelfAdjointMatrix(Args ... args)
      : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a self-adjoint matrix.
     * \note Operative if NestedMatrix is not a \ref diagonal_matrix but storage_triangle is TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    template<std::convertible_to<const Scalar> ... Args>
    requires (sizeof...(Args) > 0) and
      (storage_triangle == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      requires(Args ... args) { NestedMatrix {
        MatrixTraits<typename MatrixTraits<NestedMatrix>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and
      std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (storage_triangle == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      (std::is_constructible_v<NestedMatrix, eigen_matrix_t<Scalar, sizeof...(Args), 1>> or
       std::is_constructible_v<NestedMatrix,
         eigen_matrix_t<Scalar, constexpr_sqrt(sizeof...(Args)), constexpr_sqrt(sizeof...(Args))>>), int> = 0>
#endif
    SelfAdjointMatrix(Args ... args)
      : Base {MatrixTraits<typename MatrixTraits<NestedMatrix>::template DiagonalMatrixFrom<>>::make(
        static_cast<const Scalar>(args)...)} {}


    /** Copy assignment operator
     * \param other Another SelfAdjointMatrix
     * \return Reference to this.
     */
    auto& operator=(const SelfAdjointMatrix& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
        if (this != &other)
        {
          this->nested_matrix().template triangularView<uplo>() = other.nested_matrix();
        }
      return *this;
    }


    /** Move assignment operator
     * \param other A SelfAdjointMatrix temporary value.
     * \return Reference to this.
     */
    auto& operator=(SelfAdjointMatrix&& other) noexcept
    {
      Base::operator=(std::move(other));
      return *this;
    }


    /// Assign from another self-adjoint matrix.
#ifdef __cpp_concepts
    template<eigen_self_adjoint_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      (MatrixTraits<Arg>::rows == dimensions) and
      modifiable<NestedMatrix, nested_matrix_t<Arg>> and
      (not (eigen_diagonal_expr<NestedMatrix> and storage_triangle == TriangleType::diagonal) or diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> and
      (not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and (MatrixTraits<Arg>::rows == dimensions) and
      modifiable<NestedMatrix, nested_matrix_t<Arg>> and
      ((not eigen_diagonal_expr<NestedMatrix> and storage_triangle != TriangleType::diagonal) or diagonal_matrix<Arg>),
      int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (zero_matrix<NestedMatrix> or identity_matrix<NestedMatrix>)
      {}
      else if constexpr (eigen_diagonal_expr<NestedMatrix>)
      {
        this->nested_matrix().nested_matrix() = diagonal_of(nested_matrix(std::forward<Arg>(arg)));
      }
      else if constexpr (storage_triangle == TriangleType::diagonal)
      {
        this->nested_matrix().diagonal() = diagonal_of(nested_matrix(std::forward<Arg>(arg)));
      }
      else if constexpr (not self_adjoint_triangle_type_of_v<Arg> == self_adjoint_triangle_type_of_v<SelfAdjointMatrix>)
      {
        this->nested_matrix().template triangularView<uplo>() = transpose(nested_matrix(std::forward<Arg>(arg)));
      }
      else if constexpr(std::is_lvalue_reference_v<Arg>)
      {
        this->nested_matrix().template triangularView<uplo>() = nested_matrix(arg);
      }
      else
      {
        this->nested_matrix() = nested_matrix(std::forward<Arg>(arg));
      }
      return *this;
    }


    /// Assign from a general \ref self_adjoint_matrix.
#ifdef __cpp_concepts
    template<self_adjoint_matrix Arg> requires (not eigen_self_adjoint_expr<Arg>) and
      (MatrixTraits<Arg>::rows == dimensions) and modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<self_adjoint_matrix<Arg> and (not eigen_self_adjoint_expr<Arg>) and
      (MatrixTraits<Arg>::rows == dimensions) and modifiable<NestedMatrix, Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (zero_matrix<NestedMatrix> or identity_matrix<NestedMatrix>)
      {}
      else if constexpr (zero_matrix<Arg>)
      {
        this->nested_matrix() = MatrixTraits<NestedMatrix>::zero();
      }
      else if constexpr (identity_matrix<Arg>)
      {
        this->nested_matrix() = MatrixTraits<NestedMatrix>::identity();
      }
      else if constexpr (Eigen3::eigen_SelfAdjointView<Arg>)
      {
        if constexpr(self_adjoint_triangle_type_of_v<Arg> == self_adjoint_triangle_type_of_v<SelfAdjointMatrix>)
        {
          if constexpr (std::is_rvalue_reference_v<Arg>)
          {
            this->nested_matrix() = std::move(arg.nestedExpression());
          }
          else
          {
            this->nested_matrix().template triangularView<uplo>() = arg.nestedExpression();
          }
        }
        else
        {
          this->nested_matrix().template triangularView<uplo>() = arg.nestedExpression().transpose();
        }
      }
      else
      {
        this->nested_matrix() = std::forward<Arg>(arg);
      }
      return *this;
    }


    template<typename Arg, TriangleType t>
    auto& operator+=(const SelfAdjointMatrix<Arg, t>& arg)
    {
      static_assert(MatrixTraits<Arg>::rows == dimensions);
      if constexpr(t == storage_triangle)
        this->nested_matrix().template triangularView<uplo>() += arg.nested_matrix();
      else
        this->nested_matrix().template triangularView<uplo>() += transpose(arg.nested_matrix());
      return *this;
    }


    template<typename Arg, TriangleType t>
    auto& operator-=(const SelfAdjointMatrix<Arg, t>& arg)
    {
      static_assert(MatrixTraits<Arg>::rows == dimensions);
      if constexpr(t == storage_triangle)
        this->nested_matrix().template triangularView<uplo>() -= arg.nested_matrix();
      else
        this->nested_matrix().template triangularView<uplo>() -= transpose(arg.nested_matrix());
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      this->nested_matrix().template triangularView<uplo>() *= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      this->nested_matrix().template triangularView<uplo>() /= s;
      return *this;
    }


    auto view()
    {
      static_assert(not Eigen3::eigen_diagonal_expr<NestedMatrix>);
      return this->nested_matrix().template selfadjointView<uplo>();
    }


    const auto view() const
    {
      static_assert(not Eigen3::eigen_diagonal_expr<NestedMatrix>);
      return this->nested_matrix().template selfadjointView<uplo>();
    }

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  SelfAdjointMatrix(Arg&&) -> SelfAdjointMatrix<passable_t<Arg>, TriangleType::diagonal>;


#ifdef __cpp_concepts
  template<eigen_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  explicit SelfAdjointMatrix(Arg&&) -> SelfAdjointMatrix<passable_t<Arg>, TriangleType::lower>;


#ifdef __cpp_concepts
  template<eigen_triangular_expr Arg> requires diagonal_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg> and diagonal_matrix<Arg>, int> = 0>
#endif
  explicit SelfAdjointMatrix(Arg&&) ->
    SelfAdjointMatrix<self_contained_t<nested_matrix_t<Arg>>, TriangleType::diagonal>;


#ifdef __cpp_concepts
  template<Eigen3::eigen_SelfAdjointView M>
#else
  template<typename M, std::enable_if_t<Eigen3::eigen_SelfAdjointView<M>, int> = 0>
#endif
  SelfAdjointMatrix(M&&) -> SelfAdjointMatrix<nested_matrix_t<M>, self_adjoint_triangle_type_of_v<M>>;


  /// If the arguments are a sequence of scalars, deduce a square, self-adjoint matrix.
#ifdef __cpp_concepts
  template<arithmetic_or_complex Arg, arithmetic_or_complex ... Args> requires (std::common_with<Arg, Args> and ...)
#else
    template<typename Arg, typename ... Args, std::enable_if_t<
    (arithmetic_or_complex<Arg> and ... and arithmetic_or_complex<Args>), int> = 0>
#endif
  SelfAdjointMatrix(const Arg&, const Args& ...) -> SelfAdjointMatrix<
    Eigen3::eigen_matrix_t<
      std::common_type_t<Arg, Args...>,
      constexpr_sqrt(1 + sizeof...(Args)),
      constexpr_sqrt(1 + sizeof...(Args))>,
    TriangleType::lower>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, typename M>
  requires eigen_matrix<M> or eigen_diagonal_expr<M>
#else
  template<
    TriangleType t = TriangleType::lower, typename M,
    std::enable_if_t<eigen_matrix<M> or eigen_diagonal_expr<M>, int> = 0>
#endif
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    return SelfAdjointMatrix<passable_t<M>, t> {std::forward<M>(m)};
  }


#ifdef __cpp_concepts
  template<TriangleType t, eigen_self_adjoint_expr M>
#else
  template<TriangleType t, typename M, std::enable_if_t<eigen_self_adjoint_expr<M>, int> = 0>
#endif
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    if constexpr(t == self_adjoint_triangle_type_of_v<M>)
      return make_EigenSelfAdjointMatrix<t>(nested_matrix(std::forward<M>(m)));
    else
      return make_EigenSelfAdjointMatrix<t>(transpose(nested_matrix(std::forward<M>(m))));
  }


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr M>
#else
  template<typename M, std::enable_if_t<eigen_self_adjoint_expr<M>, int> = 0>
#endif
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    return make_EigenSelfAdjointMatrix<self_adjoint_triangle_type_of_v<M>>(std::forward<M>(m));
  }

} // OpenKalman::Eigen3


namespace OpenKalman
{
  // -------- //
  //  Traits  //
  // -------- //

  template<typename ArgType, TriangleType triangle_type>
  struct MatrixTraits<Eigen3::SelfAdjointMatrix<ArgType, triangle_type>>
  {
    static constexpr TriangleType storage_triangle = triangle_type;
    using NestedMatrix = ArgType;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto rows = MatrixTraits<NestedMatrix>::rows;
    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns;

    template<typename Derived>
    using MatrixBaseFrom =
      Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::SelfAdjointMatrix<ArgType, storage_triangle>>;

    template<std::size_t r = rows, std::size_t c = rows, typename S = Scalar>
    using NativeMatrixFrom = native_matrix_t<NestedMatrix, r, c, S>;

    using SelfContainedFrom = Eigen3::SelfAdjointMatrix<self_contained_t<NestedMatrix>, storage_triangle>;

    template<TriangleType t = storage_triangle, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<NativeMatrixFrom<dim, dim, S>, t>;

    template<TriangleType t = storage_triangle, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<NativeMatrixFrom<dim, dim, S>, t>;

    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<NativeMatrixFrom<dim, 1, S>>;


#ifdef __cpp_concepts
    template<TriangleType t = storage_triangle, typename Arg> requires
      (Eigen3::eigen_matrix<Arg> or Eigen3::eigen_diagonal_expr<Arg>)
#else
    template<TriangleType t = storage_triangle, typename Arg, std::enable_if_t<
      Eigen3::eigen_matrix<Arg> or Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Eigen3::SelfAdjointMatrix<Arg, t> {std::forward<Arg>(arg)};
    }


#ifdef __cpp_concepts
    template<TriangleType t = storage_triangle, diagonal_matrix Arg> requires
      Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>
#else
    template<TriangleType t = storage_triangle, typename Arg, std::enable_if_t<diagonal_matrix<Arg> and
      (Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Eigen3::SelfAdjointMatrix<nested_matrix_t<Arg>, t> {std::forward<Arg>(arg)};
    }


    /// Make self-adjoint matrix using a list of coefficients in row-major order.
    /// Only the coefficients in the lower-left corner are significant.
#ifdef __cpp_concepts
    template<TriangleType t = storage_triangle, std::convertible_to<Scalar> ... Args>
#else
    template<TriangleType t = storage_triangle, typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...>, int> = 0>
#endif
    static auto make(const Args ... args)
    {
      return make<t>(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return MatrixTraits<NestedMatrix>::zero(static_cast<std::size_t>(args)...);
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      return MatrixTraits<NestedMatrix>::identity(args...);
    }

  };

} // namespace OpenKalman


#endif //OPENKALMAN_EIGEN3_SELFADJOINTMATRIX_HPP

