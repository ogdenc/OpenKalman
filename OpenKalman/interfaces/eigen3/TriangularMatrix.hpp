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
 * \brief Definitions for Eigen3::TriangularMatrix
 */

#ifndef OPENKALMAN_EIGEN3_TRIANGULARMATRIX_HPP
#define OPENKALMAN_EIGEN3_TRIANGULARMATRIX_HPP

namespace OpenKalman::Eigen3
{
#ifdef __cpp_concepts
  template<typename NestedMatrix, TriangleType triangle_type> requires
    (eigen_matrix<NestedMatrix> or eigen_diagonal_expr<NestedMatrix>)
#else
  template<typename NestedMatrix, TriangleType triangle_type>
#endif
  struct TriangularMatrix
    : OpenKalman::internal::MatrixBase<TriangularMatrix<NestedMatrix, triangle_type>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(eigen_matrix<NestedMatrix> or eigen_diagonal_expr<NestedMatrix>);
#endif

    static_assert(dynamic_shape<NestedMatrix> or square_matrix<NestedMatrix>);

  private:

    using Base = OpenKalman::internal::MatrixBase<TriangularMatrix, NestedMatrix>;

    static constexpr auto uplo = triangle_type == TriangleType::upper ? Eigen::Upper : Eigen::Lower;

    static constexpr auto dimensions = MatrixTraits<NestedMatrix>::rows;

  public:

    using Scalar = typename Base::Scalar;


    /// Default constructor.
#ifdef __cpp_concepts
    TriangularMatrix() requires std::default_initializable<Base>
#else
    template<typename T = Base, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    TriangularMatrix()
#endif
      : Base {} {}


    /// Copy constructor.
    TriangularMatrix(const TriangularMatrix& other) : Base {other} {}


    /// Move constructor.
    TriangularMatrix(TriangularMatrix&& other) noexcept : Base {std::move(other)} {}


    /// Construct from a compatible triangular matrix object of the same TriangleType.
#ifdef __cpp_concepts
    template<typename  Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      ((eigen_triangular_expr<Arg> and OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix>) or
       (eigen_self_adjoint_expr<Arg> and diagonal_matrix<Arg>)) and
      (not eigen_diagonal_expr<NestedMatrix> or diagonal_matrix<nested_matrix_t<Arg>>) and
      requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
      ((eigen_triangular_expr<Arg> and OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix>) or
       (eigen_self_adjoint_expr<Arg> and diagonal_matrix<Arg>)) and
      (not eigen_diagonal_expr<NestedMatrix> or diagonal_matrix<nested_matrix_t<Arg>>) and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


      /// Construct from a compatible triangular matrix object if NestedMatrix is \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
      template<typename Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
        ((eigen_triangular_expr<Arg> and OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix>) or
         (eigen_self_adjoint_expr<Arg> and diagonal_matrix<Arg>)) and
        eigen_diagonal_expr<NestedMatrix> and (not diagonal_matrix<nested_matrix_t<Arg>>) and
        requires(Arg&& arg) { NestedMatrix {to_diagonal(diagonal_of(nested_matrix(std::forward<Arg>(arg))))}; }
#else
      template<typename Arg, std::enable_if_t<(not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
        ((eigen_triangular_expr<Arg> and OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix>) or
         (eigen_self_adjoint_expr<Arg> and diagonal_matrix<Arg>)) and
        eigen_diagonal_expr<NestedMatrix> and (not diagonal_matrix<nested_matrix_t<Arg>>) and
        std::is_constructible_v<NestedMatrix, decltype(to_diagonal(diagonal_of(nested_matrix(std::declval<Arg&&>()))))>,
        int> = 0>
#endif
      TriangularMatrix(Arg&& arg) noexcept : Base {to_diagonal(diagonal_of(nested_matrix(std::forward<Arg>(arg))))} {}


      /// Construct from an \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_diagonal_expr Arg> requires std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<
      eigen_diagonal_expr<Arg> and std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from triangular Eigen::TriangularBase-derived object.
#ifdef __cpp_concepts
    template<triangular_matrix Arg> requires
      std::derived_from<std::decay_t<Arg>, Eigen::TriangularBase<std::decay_t<Arg>>> and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and
      requires(Arg&& arg) { NestedMatrix {arg.nestedExpression()}; }
#else
    template<typename Arg, std::enable_if_t<triangular_matrix<Arg> and
      std::is_base_of_v<Eigen::TriangularBase<std::decay_t<Arg>>, std::decay_t<Arg>> and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and
      std::is_constructible_v<NestedMatrix, decltype(std::declval<Arg&&>().nestedExpression())>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) : Base {arg.nestedExpression()} {}


    /// Construct from a \ref eigen_matrix if NestedMatrix is \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires eigen_diagonal_expr<NestedMatrix> and square_matrix<Arg> and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<
      eigen_matrix<Arg> and eigen_diagonal_expr<NestedMatrix> and square_matrix<Arg> and
      std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>, int> = 0>
#endif
    explicit TriangularMatrix(Arg&& arg) noexcept : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /// Construct from a \ref eigen_matrix if NestedMatrix is not \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (not eigen_diagonal_expr<NestedMatrix>) and
      square_matrix<Arg> and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (not eigen_diagonal_expr<NestedMatrix>) and
      square_matrix<Arg> and std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit TriangularMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a triangular matrix.
     * \note Operative if triangle_type is not TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      (triangle_type != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      requires(Args ... args) { NestedMatrix {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and (triangle_type != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      (std::is_constructible_v<NestedMatrix,
          eigen_matrix_t<Scalar, OpenKalman::internal::constexpr_sqrt(sizeof...(Args)),
          OpenKalman::internal::constexpr_sqrt(sizeof...(Args))>> or
        (diagonal_matrix<NestedMatrix> and std::is_constructible_v<NestedMatrix,
          eigen_matrix_t<Scalar, sizeof...(Args), 1>>)), int> = 0>
#endif
    TriangularMatrix(Args ... args)
      : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a triangular matrix.
     * \note Operative if NestedMatrix is not \ref eigen_diagonal_expr but triangle_type is
     * TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#if defined (__cpp_concepts) && defined (__clang__) // Because of compiler issue in at least GCC version 10.1.0
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      (triangle_type == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      requires(Args ... args) { NestedMatrix {
        MatrixTraits<typename MatrixTraits<NestedMatrix>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and (triangle_type == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      (std::is_constructible_v<NestedMatrix, eigen_matrix_t<Scalar, sizeof...(Args), 1>> or
       std::is_constructible_v<NestedMatrix,
         eigen_matrix_t<Scalar, OpenKalman::internal::constexpr_sqrt(sizeof...(Args)),
         OpenKalman::internal::constexpr_sqrt(sizeof...(Args))>>), int> = 0>
#endif
    TriangularMatrix(Args ... args)
      : Base {MatrixTraits<typename MatrixTraits<NestedMatrix>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(args)...)} {}


    /// Copy assignment operator
    TriangularMatrix& operator=(const TriangularMatrix& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
        if (this != &other)
        {
          this->nested_matrix().template triangularView<uplo>() = other.nested_matrix();
        }
      return *this;
    }


    /// Move assignment operator
    TriangularMatrix& operator=(TriangularMatrix&& other) noexcept
    {
      Base::operator=(std::move(other));
      return *this;
    }


    /// Assign from another triangular matrix (must be the same triangle)
#ifdef __cpp_concepts
    template<eigen_triangular_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      (MatrixTraits<Arg>::rows == dimensions) and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and
      modifiable<NestedMatrix, nested_matrix_t<Arg>> and
      (not (eigen_diagonal_expr<NestedMatrix> or triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg> and
      (not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
      (MatrixTraits<Arg>::rows == dimensions) and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and
      modifiable<NestedMatrix, nested_matrix_t<Arg>> and
      (not (eigen_diagonal_expr<NestedMatrix> and triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>),
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
      else if constexpr (triangle_type == TriangleType::diagonal)
      {
        this->nested_matrix().diagonal() = diagonal_of(nested_matrix(std::forward<Arg>(arg)));
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


    /// Assign from a general \ref triangular_matrix.
#ifdef __cpp_concepts
    template<triangular_matrix Arg> requires (not eigen_triangular_expr<Arg>) and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and (MatrixTraits<Arg>::rows == dimensions) and
      modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<triangular_matrix<Arg> and (not eigen_triangular_expr<Arg>) and
      OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrix> and (MatrixTraits<Arg>::rows == dimensions) and
      modifiable<NestedMatrix, Arg>, int> = 0>
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
      else if constexpr (std::is_base_of_v<Eigen::TriangularBase<std::decay_t<Arg>>, std::decay_t<Arg>>)
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
        this->nested_matrix() = std::forward<Arg>(arg);
      }
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<MatrixTraits<Arg>::rows == dimensions, int> = 0>
#endif
    auto& operator+=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      view() += arg.nested_matrix();
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<MatrixTraits<Arg>::rows == dimensions, int> = 0>
#endif
    auto& operator-=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      view() -= arg.nested_matrix();
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      view() *= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      view() /= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (MatrixTraits<Arg>::rows == dimensions)
#else
    template<typename Arg, std::enable_if_t<MatrixTraits<Arg>::rows == dimensions, int> = 0>
#endif
    auto& operator*=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      auto v {view()};
      v = v * make_native_matrix(arg);
      return *this;
    }


    auto operator()(std::size_t i, std::size_t j)
    {
      if constexpr (element_settable<TriangularMatrix, 2>)
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
      if constexpr(element_gettable<TriangularMatrix,1>)
        return OpenKalman::internal::ElementAccessor(*this, i);
      else if constexpr(element_gettable<TriangularMatrix, 2>)
        return OpenKalman::internal::ElementAccessor(*this, i, i);
      else
        return std::as_const(*this)[i];
    }


    auto operator[](std::size_t i) const
    {
      if constexpr(element_gettable<TriangularMatrix, 1>)
        return OpenKalman::internal::ElementAccessor(*this, i);
      else
        return OpenKalman::internal::ElementAccessor(*this, i, i);
    }


    auto operator()(std::size_t i) { return operator[](i); }


    auto operator()(std::size_t i) const { return operator[](i); }


    auto view()
    {
      return this->nested_matrix().template triangularView<uplo>();
    }


    const auto view() const
    {
      return this->nested_matrix().template triangularView<uplo>();
    }


  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<eigen_diagonal_expr M>
#else
  template<typename M, std::enable_if_t<eigen_diagonal_expr<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<passable_t<M>, TriangleType::diagonal>;


#ifdef __cpp_concepts
  template<eigen_matrix M>
#else
  template<typename M, std::enable_if_t<eigen_matrix<M>, int> = 0>
#endif
  explicit TriangularMatrix(M&&) -> TriangularMatrix<passable_t<M>, TriangleType::lower>;


  template<typename Arg, unsigned int UpLo>
  TriangularMatrix(const Eigen::TriangularView<Arg, UpLo>&)
  -> TriangularMatrix<Arg, UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower>;


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr M> requires diagonal_matrix<M>
#else
  template<typename M, std::enable_if_t<eigen_self_adjoint_expr<M> and diagonal_matrix<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<self_contained_t<nested_matrix_t<M>>, TriangleType::diagonal>;


  /// If the arguments are a sequence of scalars, deduce a square, lower triangular matrix.
#ifdef __cpp_concepts
  template<typename Arg, typename ... Args> requires
    (std::is_arithmetic_v<std::decay_t<Arg>> and ... and std::is_arithmetic_v<std::decay_t<Args>>) and
    (std::common_with<Arg, Args> and ...)
#else
    template<typename Arg, typename ... Args, std::enable_if_t<
    (std::is_arithmetic_v<std::decay_t<Arg>> and ... and std::is_arithmetic_v<std::decay_t<Args>>), int> = 0>
#endif
  TriangularMatrix(Arg, Args ...) -> TriangularMatrix<
    Eigen::Matrix<
      std::common_type_t<std::decay_t<Arg>, std::decay_t<Args>...>,
      OpenKalman::internal::constexpr_sqrt(1 + sizeof...(Args)),
      OpenKalman::internal::constexpr_sqrt(1 + sizeof...(Args))>,
    TriangleType::lower>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, typename M> requires eigen_matrix<M> or eigen_diagonal_expr<M>
#else
  template<
    TriangleType t = TriangleType::lower, typename M,
    std::enable_if_t<eigen_matrix<M> or eigen_diagonal_expr<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return TriangularMatrix<passable_t<M>, t> (std::forward<M>(m));
  }


#ifdef __cpp_concepts
  template<TriangleType t, eigen_triangular_expr M> requires (t == MatrixTraits<M>::triangle_type)
#else
  template<TriangleType t, typename M, std::enable_if_t<eigen_triangular_expr<M> and
    (t == MatrixTraits<M>::triangle_type), int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return make_EigenTriangularMatrix<t>(nested_matrix(std::forward<M>(m)));
  }


#ifdef __cpp_concepts
  template<eigen_triangular_expr M>
#else
  template<typename M, std::enable_if_t<eigen_triangular_expr<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return make_EigenTriangularMatrix<MatrixTraits<M>::triangle_type>(nested_matrix(std::forward<M>(m)));
  }

} // namespace OpenKalman::Eigen3


namespace OpenKalman
{
  // --------------------- //
  //        Traits         //
  // --------------------- //

  template<typename ArgType, TriangleType triangle>
  struct MatrixTraits<Eigen3::TriangularMatrix<ArgType, triangle>>
  {
    static constexpr TriangleType triangle_type = triangle;
    using NestedMatrix = ArgType;
    static_assert(square_matrix<NestedMatrix>);
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto rows = MatrixTraits<NestedMatrix>::rows;
    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns;

    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::TriangularMatrix<ArgType, triangle>>;

    template<std::size_t r = rows, std::size_t c = rows, typename S = Scalar>
    using NativeMatrixFrom = native_matrix_t<NestedMatrix, r, c, S>;

    using SelfContainedFrom = Eigen3::TriangularMatrix<self_contained_t<NestedMatrix>, triangle_type>;

    template<TriangleType t = triangle_type, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<NativeMatrixFrom<dim, dim, S>, t>;

    template<TriangleType t = triangle_type, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<NativeMatrixFrom<dim, dim, S>, t>;

    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<NativeMatrixFrom<dim, 1, S>>;


#ifdef __cpp_concepts
    template<TriangleType t = triangle_type, typename Arg> requires
      Eigen3::eigen_matrix<Arg> or Eigen3::eigen_diagonal_expr<Arg>
#else
    template<TriangleType t = triangle_type, typename Arg, std::enable_if_t<
      Eigen3::eigen_matrix<Arg> or Eigen3::eigen_diagonal_expr<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Eigen3::TriangularMatrix<Arg, t> {std::forward<Arg>(arg)};
    }


#ifdef __cpp_concepts
    template<TriangleType t = triangle_type, diagonal_matrix Arg> requires Eigen3::eigen_self_adjoint_expr<Arg> or
      (Eigen3::eigen_triangular_expr<Arg> and OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrixFrom<t>>)
#else
    template<TriangleType t = triangle_type, typename Arg, std::enable_if_t<
      diagonal_matrix<Arg> and (Eigen3::eigen_self_adjoint_expr<Arg> or
      (Eigen3::eigen_triangular_expr<Arg> and
          OpenKalman::internal::same_triangle_type_as<Arg, TriangularMatrixFrom<t>>)), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Eigen3::TriangularMatrix<nested_matrix_t<Arg>, t> {std::forward<Arg>(arg)};
    }


    /// Make triangular matrix using a list of coefficients in row-major order.
    /// Only the coefficients in the lower-left corner are significant.
#ifdef __cpp_concepts

    template<TriangleType t = triangle_type, std::convertible_to<Scalar> ... Args>
#else
    template<TriangleType t = triangle_type, typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Scalar>...>, int> = 0>
#endif
    static auto make(const Args...args)
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


#endif //OPENKALMAN_EIGEN3_TRIANGULARMATRIX_HPP

