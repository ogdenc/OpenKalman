/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_SQUAREROOTCOVARIANCE_H
#define OPENKALMAN_SQUAREROOTCOVARIANCE_H

#include <initializer_list>
#include "variables/support/CovarianceBase.h"

namespace OpenKalman
{
  template<typename Coeffs, typename ArgType>
  struct SquareRootCovariance
    : internal::CovarianceBase<SquareRootCovariance<Coeffs, ArgType>, ArgType>
  {
    using BaseMatrix = ArgType;
    using Coefficients = Coeffs;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    using Base = internal::CovarianceBase<SquareRootCovariance, ArgType>;
    static constexpr TriangleType triangle_type =
      OpenKalman::triangle_type_of_v<typename MatrixTraits<BaseMatrix>::template TriangularBaseType<>>;

  protected:
    using TBaseType = std::conditional_t<is_diagonal_v<BaseMatrix>, BaseMatrix,
      typename MatrixTraits<BaseMatrix>::template TriangularBaseType<triangle_type>>;

    template<typename C = Coefficients, typename Arg>
    static constexpr auto
    make(Arg&& arg) noexcept
    {
      return SquareRootCovariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

  public:
    /**************
     * Constructors
     **************/

    /// Default constructor.
    SquareRootCovariance() : Base() {}

    /// Copy constructor.
    SquareRootCovariance(const SquareRootCovariance& other) : Base(other.base_matrix()) {}

    /// Move constructor.
    SquareRootCovariance(SquareRootCovariance&& other) : Base(std::move(other).base_matrix()) {}

    /// Construct from a general covariance type.
    template<typename M, std::enable_if_t<is_covariance_v<M>, int> = 0>
    SquareRootCovariance(M&& m) noexcept : Base(internal::convert_base_matrix<BaseMatrix>(std::forward<M>(m)))
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<M>::Coefficients, Coefficients>);
      using MBase = typename MatrixTraits<M>::BaseMatrix;
      static_assert(not is_square_root_v<M> or is_self_adjoint_v<MBase> or is_self_adjoint_v<BaseMatrix> or
          is_upper_triangular_v<BaseMatrix> == is_upper_triangular_v<MBase>,
        "An upper-triangle Cholesky-form covariance cannot be constructed from a lower-triangle Cholesky-form "
        "covariance, and vice versa. To convert, use adjoint().");
    }

    /// Construct from a non-diagonal covariance base.
    template<typename M, std::enable_if_t<is_covariance_base_v<M> and
      (not is_diagonal_v<M> or is_triangular_v<BaseMatrix>), int> = 0>
    SquareRootCovariance(M&& m) noexcept : Base(std::forward<M>(m)) {}

    /// Construct from a diagonal covariance base.
    template<typename M, std::enable_if_t<is_covariance_base_v<M> and
      is_diagonal_v<M> and not is_triangular_v<BaseMatrix>, int> = 0>
    SquareRootCovariance(M&& m) noexcept : Base(Cholesky_square(std::forward<M>(m))) {}

    /// Construct from a typed matrix (assumed to be triangular).
    template<typename M, std::enable_if_t<is_typed_matrix_v<M>, int> = 0>
    SquareRootCovariance(M&& m) noexcept
      : Base(MatrixTraits<TBaseType>::make(OpenKalman::base_matrix(std::forward<M>(m))))
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<M>::RowCoefficients, Coefficients>);
      if constexpr(is_diagonal_v<BaseMatrix>)
        static_assert(MatrixTraits<M>::columns == 1);
      else
        static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<M>::ColumnCoefficients, Coefficients>);
    }

    /// Construct from a typed matrix base (assumed to be triangular).
    template<typename M, std::enable_if_t<
      is_typed_matrix_base_v<M> and not is_covariance_base_v<M>, int> = 0>
    SquareRootCovariance(M&& m) noexcept : Base(MatrixTraits<TBaseType>::make(std::forward<M>(m))) {}

    /// Construct from Scalar coefficients. Assumes matrix is triangular, and only reads lower left triangle.
    template<typename ... Args, std::enable_if_t<
      std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
    SquareRootCovariance(Args ... args) : Base(MatrixTraits<TBaseType>::make(args...)) {}

    /**********************
     * Assignment Operators
     **********************/

    /// Copy assignment operator.
    auto& operator=(const SquareRootCovariance& other)
    {
      if (this != &other) base_matrix() = other.base_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(SquareRootCovariance&& other) noexcept
    {
      if (this != &other) base_matrix() = std::move(other).base_matrix();
      return *this;
    }

    /// Assign from a compatible covariance object.
    template<typename Arg, std::enable_if_t<is_covariance_v<Arg>, int> = 0>
    auto& operator=(Arg&& other) noexcept
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      using ArgBase = typename MatrixTraits<Arg>::BaseMatrix;
      static_assert(not is_square_root_v<Arg> or is_self_adjoint_v<ArgBase> or is_self_adjoint_v<BaseMatrix> or
        is_upper_triangular_v<BaseMatrix> == is_upper_triangular_v<ArgBase>,
          "An upper-triangle Cholesky-form covariance cannot be assigned a lower-triangle Cholesky-form "
          "covariance, and vice versa. To convert, use adjoint().");
      base_matrix() = internal::convert_base_matrix<BaseMatrix>(std::forward<Arg>(other));
      return *this;
    }

    /// Warning: This is computationally expensive if the base matrix is self-adjoint.
    template<typename Arg, std::enable_if_t<is_covariance_v<Arg>, int> = 0>
    auto& operator+=(Arg&& arg)
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      static_assert(OpenKalman::is_square_root_v<Arg>);
      static_assert(is_upper_triangular_v<SquareRootCovariance> == OpenKalman::is_upper_triangular_v<Arg>);
      if constexpr(OpenKalman::is_triangular_v<BaseMatrix>)
      {
        base_matrix() += internal::convert_base_matrix<BaseMatrix>(std::forward<Arg>(arg));
      }
      else
      {
        const auto sum = internal::convert_base_matrix<TBaseType>(*this) +
          internal::convert_base_matrix<TBaseType>(std::forward<Arg>(arg));
        base_matrix() = Cholesky_square(sum);
      }
      return *this;
    }

    /// Warning: This is computationally expensive if the base matrix is self-adjoint.
    template<typename Arg, std::enable_if_t<is_covariance_v<Arg>, int> = 0>
    auto& operator-=(Arg&& arg)
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      static_assert(OpenKalman::is_square_root_v<Arg>);
      static_assert(is_upper_triangular_v<SquareRootCovariance> == OpenKalman::is_upper_triangular_v<Arg>);
      if constexpr(OpenKalman::is_triangular_v<BaseMatrix>)
      {
        base_matrix() -= internal::convert_base_matrix<BaseMatrix>(std::forward<Arg>(arg));
      }
      else
      {
        const auto sum = internal::convert_base_matrix<TBaseType>(*this) -
          internal::convert_base_matrix<TBaseType>(std::forward<Arg>(arg));
        base_matrix() = Cholesky_square(sum);
      }
      return *this;
    }

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator*=(const S s)
    {
      if constexpr(OpenKalman::is_triangular_v<BaseMatrix>)
      {
        base_matrix() *= s;
      }
      else
      {
        base_matrix() *= static_cast<Scalar>(s) * s;
      }
      return *this;
    }

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator/=(const S s)
    {
      if constexpr(OpenKalman::is_triangular_v<BaseMatrix>)
      {
        base_matrix() /= s;
      }
      else
      {
        base_matrix() /= static_cast<Scalar>(s) * s;
      }
      return *this;
    }

    /// Warning: This is computationally expensive unless &this and Arg are both the same triangular kind.
    template<typename Arg, std::enable_if_t<is_covariance_v<Arg>, int> = 0>
    auto& operator*=(Arg&& arg)
    {
      static_assert(is_square_root_v<Arg> and is_upper_triangular_v<SquareRootCovariance> == is_upper_triangular_v<Arg>,
        "operator*=() requires that both Cholesky-form covariances are of the same triangular kind (both upper or lower).");
      if constexpr(OpenKalman::is_triangular_v<BaseMatrix> or is_diagonal_v<BaseMatrix>)
      {
        base_matrix() *= internal::convert_base_matrix<BaseMatrix>(std::forward<Arg>(arg));
      }
      else
      {
        // Convert both operands to triangular matrices, and then back to self-adjoint.
        const auto prod = internal::convert_base_matrix<TBaseType>(*this) *
          internal::convert_base_matrix<TBaseType>(std::forward<Arg>(arg));
        base_matrix() = Cholesky_square(prod);
      }
      return *this;
    }

    /*********
     * Other
     *********/

    static auto zero() { return make(MatrixTraits<BaseMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<BaseMatrix>::identity()); }

    using Base::base_matrix;

  };


  /////////////////////////////////////
  //        Deduction guides         //
  /////////////////////////////////////

  template<typename M, std::enable_if_t<is_covariance_v<M>, int> = 0>
  SquareRootCovariance(M&&)
    -> SquareRootCovariance<
    typename MatrixTraits<M>::Coefficients,
    typename MatrixTraits<M>::BaseMatrix>;

  template<typename M, std::enable_if_t<is_typed_matrix_v<M>, int> = 0>
  SquareRootCovariance(M&&)
    -> SquareRootCovariance<
    typename MatrixTraits<M>::RowCoefficients,
    typename MatrixTraits<typename MatrixTraits<M>::BaseMatrix>::template TriangularBaseType<>>;

  template<typename M, std::enable_if_t<is_covariance_base_v<M>, int> = 0>
  SquareRootCovariance(M&&)
    -> SquareRootCovariance<
      Axes<MatrixTraits<M>::dimension>,
      std::decay_t<M>>;

  template<typename M, std::enable_if_t<is_typed_matrix_base_v<M> and not is_covariance_base_v<M>, int> = 0>
  SquareRootCovariance(M&&)
    -> SquareRootCovariance<
      Axes<MatrixTraits<M>::dimension>,
      typename MatrixTraits<M>::template TriangularBaseType<>>;


  //////////////////////
  //  Make Functions  //
  //////////////////////

  // Make from covariance base or regular matrix:

  /// Make a SquareRootCovariance, based on a covariance base or regular matrix.
  template<typename Coefficients, TriangleType...triangle_type, typename Arg,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and
      (is_covariance_base_v<Arg> or is_typed_matrix_base_v<Arg>), int> = 0>
  auto make_SquareRootCovariance(Arg&& arg) noexcept
  {
    constexpr TriangleType arg_t_type = triangle_type_of_v<typename MatrixTraits<Arg>::template TriangularBaseType<>>;
    constexpr TriangleType t_type = (arg_t_type, ... , triangle_type);

    if constexpr(is_covariance_base_v<Arg>)
    {
      if constexpr(arg_t_type != t_type and t_type != TriangleType::diagonal and not is_diagonal_v<Arg>)
      {
        auto b = adjoint(std::forward<Arg>(arg));
        return SquareRootCovariance<Coefficients, std::decay_t<decltype(b)>>(std::move(b));
      }
      else
      {
        return SquareRootCovariance<Coefficients, std::decay_t<Arg>>(std::forward<Arg>(arg));
      }
    }
    else
    {
      using B = typename MatrixTraits<Arg>::template TriangularBaseType<t_type>;
      return SquareRootCovariance<Coefficients, B> {static_cast<B>(std::forward<Arg>(arg))};
    }
  }


  /// Make an axes-only SquareRootCovariance, based on a covariance base or regular matrix.
  template<TriangleType...triangle_type, typename Arg,
    std::enable_if_t<(sizeof...(triangle_type)) <= 1 and
      (is_covariance_base_v<Arg> or is_typed_matrix_base_v<Arg>), int> = 0>
  auto make_SquareRootCovariance(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_SquareRootCovariance<C, triangle_type...>(std::forward<Arg>(arg));
  }


  /// Make a default SquareRootCovariance, based on a template type, specifying a triangle type.
  template<typename Coefficients, TriangleType triangle_type, typename Arg,
    std::enable_if_t<is_covariance_base_v<Arg> or is_typed_matrix_base_v<Arg>, int> = 0>
  auto make_SquareRootCovariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);

    using B = std::conditional_t<OpenKalman::is_diagonal_v<Arg>,
      typename MatrixTraits<Arg>::template DiagonalBaseType<>,
      std::conditional_t<OpenKalman::is_self_adjoint_v<Arg>,
        typename MatrixTraits<Arg>::template SelfAdjointBaseType<triangle_type>,
        typename MatrixTraits<Arg>::template TriangularBaseType<triangle_type>>>;

    return SquareRootCovariance<Coefficients, B>();
  }


  /// Make a default SquareRootCovariance, based on a template type.
  template<typename Coefficients, typename Arg,
    std::enable_if_t<is_covariance_base_v<Arg> or is_typed_matrix_base_v<Arg>, int> = 0>
  auto make_SquareRootCovariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    constexpr TriangleType template_type = triangle_type_of_v<typename MatrixTraits<Arg>::template TriangularBaseType<>>;

    using B = std::conditional_t<OpenKalman::is_diagonal_v<Arg>,
      typename MatrixTraits<Arg>::template DiagonalBaseType<>,
      std::conditional_t<OpenKalman::is_self_adjoint_v<Arg>,
        typename MatrixTraits<Arg>::template SelfAdjointBaseType<template_type>,
        typename MatrixTraits<Arg>::template TriangularBaseType<template_type>>>;

    return SquareRootCovariance<Coefficients, B>();
  }


  /// Make a default axes-only SquareRootCovariance, based on a template type.
  template<TriangleType triangle_type, typename Arg,
    std::enable_if_t<is_covariance_base_v<Arg> or is_typed_matrix_base_v<Arg>, int> = 0>
  auto make_SquareRootCovariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_SquareRootCovariance<C, triangle_type, Arg>();
  }


  /// Make a default axes-only SquareRootCovariance, based on a template type.
  template<typename Arg,
    std::enable_if_t<is_covariance_base_v<Arg> or is_typed_matrix_base_v<Arg>, int> = 0>
  auto make_SquareRootCovariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_SquareRootCovariance<C, Arg>();
  }


  // Make from another covariance type

  /// Make a SquareRootCovariance based on another covariance.
  template<TriangleType...triangle_type, typename Arg,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and is_covariance_v<Arg>, int> = 0>
  auto make_SquareRootCovariance(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    return make_SquareRootCovariance<C, triangle_type...>(base_matrix(std::forward<Arg>(arg)));
  }


  /// Make a default axes-only SquareRootCovariance, based on a covariance template type.
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<is_covariance_v<Arg>, int> = 0>
  auto make_SquareRootCovariance()
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    using B = typename MatrixTraits<Arg>::BaseMatrix;
    return make_SquareRootCovariance<C, triangle_type, B>();
  }


  /// Make a default axes-only SquareRootCovariance, based on a covariance template type.
  template<typename Arg, std::enable_if_t<is_covariance_v<Arg>, int> = 0>
  auto make_SquareRootCovariance()
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    using B = typename MatrixTraits<Arg>::BaseMatrix;
    return make_SquareRootCovariance<C, B>();
  }


  // Make from a typed matrix

  /// Make a SquareRootCovariance from a typed matrix.
  template<TriangleType...triangle_type, typename Arg,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and is_typed_matrix_v<Arg>, int> = 0>
  auto make_SquareRootCovariance(Arg&& arg) noexcept
  {
    static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    return make_SquareRootCovariance<C, triangle_type...>(base_matrix(std::forward<Arg>(arg)));
  }


  /// Make a default axes-only SquareRootCovariance, based on a typed matrix template type, specifying a triangle type.
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  auto make_SquareRootCovariance()
  {
    static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using B = typename MatrixTraits<Arg>::BaseMatrix;
    return make_SquareRootCovariance<C, triangle_type, B>();
  }


  /// Make a default axes-only SquareRootCovariance, based on a typed matrix template type.
  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  auto make_SquareRootCovariance()
  {
    static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using B = typename MatrixTraits<Arg>::BaseMatrix;
    return make_SquareRootCovariance<C, B>();
  }


  ////////////////////////////
  //        Traits          //
  ////////////////////////////

  template<typename Coeffs, typename ArgType>
  struct MatrixTraits<OpenKalman::SquareRootCovariance<Coeffs, ArgType>>
  {
    using BaseMatrix = std::decay_t<ArgType>;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static_assert(Coeffs::size == dimension);
    using Coefficients = Coeffs;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar; ///< Scalar type for this vector.

    static constexpr TriangleType
    triangle_type = triangle_type_of_v<typename MatrixTraits<ArgType>::template TriangularBaseType<>>;

    template<std::size_t rows = dimension, std::size_t cols = dimension, typename S = Scalar>
    using StrictMatrix = typename MatrixTraits<BaseMatrix>::template StrictMatrix<rows, cols, S>;

    /// Make SquareRootCovariance from a covariance base.
    template<typename C = Coefficients, typename Arg, std::enable_if_t<is_covariance_base_v<Arg>, int> = 0>
    static auto make(Arg&& arg) noexcept
    {
      return make_SquareRootCovariance<C>(std::forward<Arg>(arg));
    }

    static auto zero() { return SquareRootCovariance<Coefficients, std::decay_t<BaseMatrix>>::zero(); }

    static auto identity() { return SquareRootCovariance<Coefficients, std::decay_t<BaseMatrix>>::identity(); }
  };


}


#endif //OPENKALMAN_SQUAREROOTCOVARIANCE_H

