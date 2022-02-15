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
  requires (eigen_diagonal_expr<NestedMatrix> and not complex_number<scalar_type_of_t<NestedMatrix>>) or
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
      dynamic_rows<NestedMatrix> ? column_extent_of_v<NestedMatrix> : row_extent_of_v<NestedMatrix>;


  public:

    using Scalar = scalar_type_of_t<NestedMatrix>;


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


    /// Construct from a diagonal matrix if NestedMatrix is diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      diagonal_matrix<NestedMatrix> and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and
      diagonal_matrix<Arg> and diagonal_matrix<NestedMatrix> and
      std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /// Construct from a diagonal matrix if NestedMatrix is not diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      (not diagonal_matrix<NestedMatrix> or
        not requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<
      diagonal_matrix<Arg> and (not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and
      (not diagonal_matrix<NestedMatrix> or
        not std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a hermitian, non-diagonal wrapper of the same storage type
#ifdef __cpp_concepts
    template<self_adjoint_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      (not diagonal_matrix<Arg>) and has_nested_matrix<Arg> and
      (self_adjoint_triangle_type_of<Arg>::value == storage_triangle) and
      (dynamic_shape<nested_matrix_of_t<Arg>> or square_matrix<nested_matrix_of_t<Arg>>) and
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
      //alt: requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } -- not accepted in GCC 10
#else
    template<typename Arg, std::enable_if_t<
      self_adjoint_matrix<Arg> and (not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and
      (not diagonal_matrix<Arg>) and has_nested_matrix<Arg> and
      (self_adjoint_triangle_type_of<Arg>::value == storage_triangle) and
      (dynamic_shape<nested_matrix_of_t<Arg>> or square_matrix<nested_matrix_of_t<Arg>>) and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from a hermitian, non-diagonal wrapper of the opposite storage type
#ifdef __cpp_concepts
    template<self_adjoint_matrix Arg> requires (not diagonal_matrix<Arg>) and
      has_nested_matrix<Arg> and (self_adjoint_triangle_type_of<Arg>::value != storage_triangle) and
      (dynamic_shape<nested_matrix_of_t<Arg>> or square_matrix<nested_matrix_of_t<Arg>>) and
      requires(Arg&& arg) { NestedMatrix {transpose(nested_matrix(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<
      self_adjoint_matrix<Arg> and (not diagonal_matrix<Arg>) and
      has_nested_matrix<Arg> and (self_adjoint_triangle_type_of<Arg>::value != storage_triangle) and
      (dynamic_shape<nested_matrix_of_t<Arg>> or square_matrix<nested_matrix_of_t<Arg>>) and
      std::is_constructible_v<NestedMatrix, decltype(transpose(nested_matrix(std::declval<Arg&&>())))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {transpose(nested_matrix(std::forward<Arg>(arg)))} {}


    /// Construct from a hermitian matrix of the same storage type and is not a wrapper
#ifdef __cpp_concepts
    template<self_adjoint_matrix Arg> requires (not diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg>) and (self_adjoint_triangle_type_of<Arg>::value == storage_triangle) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<self_adjoint_matrix<Arg> and (not diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg>) and (self_adjoint_triangle_type_of<Arg>::value == storage_triangle) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a hermitian matrix, of the opposite storage type, that is not a wrapper
#ifdef __cpp_concepts
    template<self_adjoint_matrix Arg> requires (not diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg>) and (self_adjoint_triangle_type_of<Arg>::value != storage_triangle) and
      requires(Arg&& arg) { NestedMatrix {transpose(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<self_adjoint_matrix<Arg> and (not diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg>) and (self_adjoint_triangle_type_of<Arg>::value != storage_triangle) and
      std::is_constructible_v<NestedMatrix, decltype(transpose(std::declval<Arg&&>()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {transpose(std::forward<Arg>(arg))} {}


    /// Construct from a non-hermitian matrix if NestedMatrix is not diagonal.
#ifdef __cpp_concepts
    template<typename Arg> requires (not self_adjoint_matrix<Arg>) and (not diagonal_matrix<NestedMatrix>) and
      (dynamic_shape<Arg> or square_matrix<Arg>) and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<not self_adjoint_matrix<Arg> and not eigen_diagonal_expr<NestedMatrix> and
      (dynamic_shape<Arg> or square_matrix<Arg>) and std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit SelfAdjointMatrix(Arg&& arg) noexcept
      : Base {[](Arg&& arg) -> decltype(auto) {
        if constexpr (dynamic_rows<Arg> and dimensions != dynamic_extent) assert(row_count(arg) == dimensions);
        if constexpr (dynamic_columns<Arg> and dimensions != dynamic_extent) assert(column_count(arg) == dimensions);
        return std::forward<Arg>(arg);
      }(std::forward<Arg>(arg))} {}


    /// Construct from a non-hermitian matrix if NestedMatrix is diagonal.
#ifdef __cpp_concepts
    template<typename Arg> requires (not self_adjoint_matrix<Arg>) and diagonal_matrix<NestedMatrix> and
      (dynamic_shape<Arg> or square_matrix<Arg>) and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<(not self_adjoint_matrix<Arg>) and diagonal_matrix<NestedMatrix> and
      (dynamic_shape<Arg> or square_matrix<Arg>) and
      std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>, int> = 0>
#endif
    explicit SelfAdjointMatrix(Arg&& arg) noexcept
      : Base {[](Arg&& arg) -> decltype(auto) {
        if constexpr (dynamic_rows<Arg> and dimensions != dynamic_extent) assert(row_count(arg) == dimensions);
        if constexpr (dynamic_columns<Arg> and dimensions != dynamic_extent) assert(column_count(arg) == dimensions);
        return diagonal_of(std::forward<Arg>(arg));
      }(std::forward<Arg>(arg))} {}


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
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
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
      (row_extent_of_v<Arg> == dimensions) and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>> and
      (not (eigen_diagonal_expr<NestedMatrix> and storage_triangle == TriangleType::diagonal) or diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> and
      (not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and (row_extent_of<Arg>::value == dimensions) and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>> and
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
      else if constexpr (self_adjoint_triangle_type_of_v<Arg> != storage_triangle)
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
      (row_extent_of_v<Arg> == dimensions) and modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<self_adjoint_matrix<Arg> and (not eigen_self_adjoint_expr<Arg>) and
      (row_extent_of<Arg>::value == dimensions) and modifiable<NestedMatrix, Arg>, int> = 0>
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
        if constexpr(self_adjoint_triangle_type_of_v<Arg> == storage_triangle)
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
      static_assert(row_extent_of_v<Arg> == dimensions);
      if constexpr(t == storage_triangle)
        this->nested_matrix().template triangularView<uplo>() += arg.nested_matrix();
      else
        this->nested_matrix().template triangularView<uplo>() += transpose(arg.nested_matrix());
      return *this;
    }


    template<typename Arg, TriangleType t>
    auto& operator-=(const SelfAdjointMatrix<Arg, t>& arg)
    {
      static_assert(row_extent_of_v<Arg> == dimensions);
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
    SelfAdjointMatrix<equivalent_self_contained_t<nested_matrix_of_t<Arg>>, TriangleType::diagonal>;


#ifdef __cpp_concepts
  template<Eigen3::eigen_SelfAdjointView M>
#else
  template<typename M, std::enable_if_t<Eigen3::eigen_SelfAdjointView<M>, int> = 0>
#endif
  SelfAdjointMatrix(M&&) -> SelfAdjointMatrix<nested_matrix_of_t<M>, self_adjoint_triangle_type_of_v<M>>;


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



#endif //OPENKALMAN_EIGEN3_SELFADJOINTMATRIX_HPP

