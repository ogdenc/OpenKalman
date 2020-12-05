/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_FROMEUCLIDEANEXPR_HPP
#define OPENKALMAN_EIGEN3_FROMEUCLIDEANEXPR_HPP

namespace OpenKalman::Eigen3
{
  // -------------------------------- //
  //        FromEuclideanExpr         //
  // -------------------------------- //

  // Documentation is in EigenForwardDeclarations.h
#ifdef __cpp_concepts
  template<coefficients Coefficients, typename NestedMatrix> requires
    (MatrixTraits<NestedMatrix>::dimension == Coefficients::dimension)
#else
  template<typename Coefficients, typename NestedMatrix>
#endif
  struct FromEuclideanExpr
    : OpenKalman::internal::MatrixBase<FromEuclideanExpr<Coefficients, NestedMatrix>, NestedMatrix>
  {
    static_assert(MatrixTraits<NestedMatrix>::dimension == Coefficients::dimension);
    using Base = OpenKalman::internal::MatrixBase<FromEuclideanExpr, NestedMatrix>;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar; ///< Scalar type for this variable.
    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns; ///< Number of columns.

    /// Default constructor.
    FromEuclideanExpr() : Base() {}

    /// Copy constructor.
    FromEuclideanExpr(const FromEuclideanExpr& other) : Base(other.nested_matrix()) {}

    /// Move constructor.
    FromEuclideanExpr(FromEuclideanExpr&& other) noexcept: Base(std::move(other).nested_matrix()) {}


    /**
     * Convert from a compatible from-euclidean expression.
     */
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr < Arg>, int> = 0>
#endif
    FromEuclideanExpr(Arg&& other) noexcept: Base(std::forward<Arg>(other).nested_matrix())
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::columns == columns);
    }


    /**
     * Construct from a compatible to-euclidean expression.
     */
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg>, int> = 0>
#endif
    FromEuclideanExpr(Arg&& other) noexcept: Base(std::forward<Arg>(other))
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::columns == columns);
    }


    /**
     * Construct from compatible matrix object.
     */
#ifdef __cpp_concepts
    template<eigen_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
    FromEuclideanExpr(Arg&& arg) noexcept : Base(std::forward<Arg>(arg))
    {
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::dimension);
      static_assert(MatrixTraits<Arg>::columns == columns);
    }


    /**
     * Construct from a list of coefficients.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires
    (sizeof...(Args) == columns * (to_euclidean_expr<NestedMatrix> ? Coefficients::size : Coefficients::dimension))
#else
    template<
      typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
        sizeof...(Args) == columns *
          (to_euclidean_expr < NestedMatrix > ? Coefficients::size : Coefficients::dimension), int> = 0>
#endif
    FromEuclideanExpr(Args ... args) : Base(MatrixTraits<NestedMatrix>::make(args...)) {}


    /// Copy assignment operator.
    auto& operator=(const FromEuclideanExpr& other)
    {
      if constexpr (not zero_matrix < NestedMatrix > and not identity_matrix<NestedMatrix>)
        if (this != &other)
          this->nested_matrix() = other.nested_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(FromEuclideanExpr&& other) noexcept
    {
      if constexpr (not zero_matrix < NestedMatrix > and not identity_matrix<NestedMatrix>)
        if (this != &other)
          this->nested_matrix() = std::move(other).nested_matrix();
      return *this;
    }

    /**
     * Assign from a compatible from-Euclidean expression.
     */
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr < Arg>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      static_assert(equivalent_to < typename MatrixTraits<Arg>::RowCoefficients, Coefficients > );
      static_assert(MatrixTraits<Arg>::columns == columns);
      if constexpr (zero_matrix < NestedMatrix >)
      {
        static_assert(zero_matrix < Arg > );
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else
      {
        this->nested_matrix() = std::forward<Arg>(other).nested_matrix();
      }
      return *this;
    }


    /**
     * Assign from a general Eigen matrix.
     */
#ifdef __cpp_concepts
    template<eigen_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::size);
      static_assert(MatrixTraits<Arg>::columns == columns);
      if constexpr (zero_matrix < NestedMatrix >)
      {
        static_assert(zero_matrix < Arg > );
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else
      {
        this->nested_matrix() = to_euclidean<Coefficients>(std::forward<Arg>(arg));
      }
      return *this;
    }


    /**
     * Increment from another expression.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires from_euclidean_expr<Arg> or eigen_matrix<Arg>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr < Arg> or eigen_matrix<Arg>, int> = 0>
#endif
    auto& operator+=(Arg&& other) noexcept
    {
      static_assert(MatrixTraits<Arg>::columns == MatrixTraits<NestedMatrix>::columns);
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::size);
      if constexpr(from_euclidean_expr < Arg >)
        static_assert(equivalent_to < typename MatrixTraits<Arg>::RowCoefficients, Coefficients > );
      this->nested_matrix() = make_self_contained(to_euclidean<Coefficients>(*this + std::forward<Arg>(other)));
      return *this;
    }


    /**
     * Decrement from another expression.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires from_euclidean_expr<Arg> or eigen_matrix<Arg>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr < Arg> or eigen_matrix<Arg>, int> = 0>
#endif
    auto& operator-=(Arg&& other) noexcept
    {
      static_assert(MatrixTraits<Arg>::columns == MatrixTraits<NestedMatrix>::columns);
      static_assert(MatrixTraits<Arg>::dimension == Coefficients::size);
      if constexpr(from_euclidean_expr < Arg >)
        static_assert(equivalent_to < typename MatrixTraits<Arg>::RowCoefficients, Coefficients > );
      this->nested_matrix() = make_self_contained(to_euclidean<Coefficients>(*this - std::forward<Arg>(other)));
      return *this;
    }


    /**
     * Multiply by a scale factor.
     * \param scale The scale factor.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S scale)
    {
      this->nested_matrix() = make_self_contained(to_euclidean<Coefficients>(*this * scale));
      return *this;
    }


    /**
     * Divide by a scale factor.
     * \param scale The scale factor.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S scale)
    {
      this->nested_matrix() = make_self_contained(to_euclidean<Coefficients>(*this / scale));
      return *this;
    }

    /**
     * A matrix with the same form factor, but with all zero coefficients.
     * \return
     */
    static auto zero()
    {
      using ST = native_matrix_t<NestedMatrix, Coefficients::size>;
      return MatrixTraits<ST>::zero();
    }

    /**
     * An identity matrix.
     * Note: <code>make_native_matrix(*this)</code> must be a square matrix.
     * \return An identity matrix.
     */
    static auto identity()
    {
      static_assert(MatrixTraits<NestedMatrix>::dimension == columns,
        "Cannot call the identity() function on a non-square matrix.");
      using ST = native_matrix_t<NestedMatrix, Coefficients::size>;
      return MatrixTraits<ST>::identity();
    }

    /**
     * Access the coefficient at row i and column j
     * \param i The row.
     * \param j The column.
     * \return If <code>element_settable<FromEuclideanExpr, 2></code>, the element is settable. Therefore,
     * this function returns an object that can be assigned the coefficient to be set.
     * Otherwise, it will return the (non-settable) coefficient as a value.
     */
    auto operator()(std::size_t i, std::size_t j)
    {
      if constexpr (element_settable < FromEuclideanExpr, 2 >)
        return OpenKalman::internal::ElementSetter(*this, i, j);
      else
        return const_cast<const FromEuclideanExpr&>(*this)(i, j);
    }

    /**
     * Access the coefficient at row i and column j
     * \param i The row.
     * \param j The column.
     * \return The value of the coefficient.
     */
    auto operator()(std::size_t i, std::size_t j) const { return OpenKalman::internal::ElementSetter(*this, i, j); }

    /**
     * Access the coefficient at row i
     * \param i The row.
     * \return If <code>element_settable<FromEuclideanExpr, 1></code>, the element is settable. Therefore,
     * this function returns an object that can be assigned the coefficient to be set.
     * Otherwise, it will return the (non-settable) coefficient as a value.
     */
    auto operator[](std::size_t i)
    {
      if constexpr (element_settable < FromEuclideanExpr, 1 >)
        return OpenKalman::internal::ElementSetter(*this, i);
      else
        return const_cast<const FromEuclideanExpr&>(*this)[i];
    }

    /**
     * Access the coefficient at row i
     * \param i The row.
     * \return The value of the coefficient.
     */
    auto operator[](std::size_t i) const { return OpenKalman::internal::ElementSetter(*this, i); }

    /**
     * Synonym for operator[](std::size_t)
     * \param i The row.
     * \return If <code>element_settable<FromEuclideanExpr, 1></code>, the element is settable. Therefore,
     * this function returns an object that can be assigned the coefficient to be set.
     * Otherwise, it will return the (non-settable) coefficient as a value.
     */
    auto operator()(std::size_t i) { return operator[](i); }

    /**
     * Synonym for operator[](std::size_t) const.
     * \param i The row.
     * \return The value of the coefficient.
     */
    auto operator()(std::size_t i) const { return operator[](i); }
  };

} // OpenKalman::Eigen3

namespace OpenKalman
{
  // --------------------- //
  //        Traits         //
  // --------------------- //

  template<typename Coeffs, typename ArgType>
  struct MatrixTraits<Eigen3::FromEuclideanExpr<Coeffs, ArgType>>
  {
    using NestedMatrix = ArgType;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto dimension = Coeffs::size;
    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns;
    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    static_assert(Coeffs::dimension == MatrixTraits<NestedMatrix>::dimension);

    template<typename Derived>
    using MatrixBaseType = Eigen3::internal::Eigen3MatrixBase<Derived, Eigen3::FromEuclideanExpr<Coeffs, ArgType>>;

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using NativeMatrix = typename MatrixTraits<NestedMatrix>::template NativeMatrix<rows, cols, S>;

    using SelfContained = Eigen3::FromEuclideanExpr<Coeffs, self_contained_t<NestedMatrix>>;

    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointBaseType = Eigen3::SelfAdjointMatrix<NativeMatrix<dim, dim, S>, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = dimension, typename S = Scalar>
    using TriangularBaseType = Eigen3::TriangularMatrix<NativeMatrix<dim, dim, S>, triangle_type>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalBaseType = Eigen3::DiagonalMatrix<NativeMatrix<dim, 1, S>>;

    // Make from a regular matrix.
#ifdef __cpp_concepts
    template<typename C = Coeffs, typename Arg> requires
      Eigen3::eigen_matrix<Arg> or Eigen3::to_euclidean_expr<Arg>
#else
    template<typename C = Coeffs, typename Arg,
      std::enable_if_t<Eigen3::eigen_matrix<Arg> or Eigen3::to_euclidean_expr<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == C::dimension);
      using namespace Eigen3;
      return from_euclidean<C>(std::forward<Arg>(arg));
    }

#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args>
#else
    template<typename ... Args, std::enable_if_t<
      std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
#endif
    static auto
    make(Args ... args)
    {
      static_assert(sizeof...(Args) == Coeffs::dimension * columns);
      return make(MatrixTraits<NestedMatrix>::make(args...));
    }

    static auto zero() { return Eigen3::FromEuclideanExpr<Coeffs, NestedMatrix>::zero(); }

    static auto identity() { return Eigen3::FromEuclideanExpr<Coeffs, NestedMatrix>::identity(); }

  };

} // namespace OpenKalman




#endif //OPENKALMAN_EIGEN3_FROMEUCLIDEANEXPR_HPP
