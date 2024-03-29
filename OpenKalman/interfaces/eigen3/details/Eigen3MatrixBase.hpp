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
 * \internal
 * \file
 * \brief Definitions for Eigen3::Eigen3MatrixBase and extensions to Eigen::CommaInitializer
 */

#ifndef OPENKALMAN_EIGEN3MATRIXBASE_HPP
#define OPENKALMAN_EIGEN3MATRIXBASE_HPP

namespace Eigen
{
  template<typename Derived, typename XprType>
  struct MeanCommaInitializer;

  template<typename XprType>
  struct DiagonalCommaInitializer;

  template<typename CovarianceType>
  struct CovarianceCommaInitializer;
}


namespace OpenKalman::Eigen3::internal
{
  /*
   * Implementation
   */
  template<typename Derived, typename ArgType>
  struct Eigen3MatrixBase : Eigen3Base<Derived>
  {
    using Nested = ArgType; ///< Required by Eigen3.

    using Scalar = typename MatrixTraits<Nested>::Scalar; ///< Required by Eigen3.

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(bool {dynamic_shape<ArgType>})

  private:

    using Base = Eigen3Base<Derived>;

    template<typename Arg>
    auto& get_ultimate_nested_matrix_impl(Arg& arg)
    {
      auto& b = nested_matrix(arg);
      using B = decltype(b);
      static_assert(not std::is_const_v<std::remove_reference_t<B>>);
      if constexpr(Eigen3::eigen_self_adjoint_expr<B> or Eigen3::eigen_triangular_expr<B> or
        Eigen3::eigen_diagonal_expr<B> or Eigen3::euclidean_expr<B>)
      {
        return get_ultimate_nested_matrix(b);
      }
      else
      {
        return b;
      }
    }


    template<typename Arg>
    auto& get_ultimate_nested_matrix(Arg& arg)
    {
      if constexpr(Eigen3::eigen_self_adjoint_expr<Arg>)
      {
        if constexpr (MatrixTraits<Arg>::storage_triangle == TriangleType::diagonal) return arg;
        else return get_ultimate_nested_matrix_impl(arg);
      }
      else if constexpr(Eigen3::eigen_triangular_expr<Arg>)
      {
        if constexpr(MatrixTraits<Arg>::triangle_type == TriangleType::diagonal) return arg;
        else return get_ultimate_nested_matrix_impl(arg);
      }
      else
      {
        return get_ultimate_nested_matrix_impl(arg);
      }
    }


  public:

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto operator<<(const S& s)
    {
      if constexpr(covariance<Derived>)
      {
        auto& xpr = static_cast<Derived&>(*this);
        return Eigen::CovarianceCommaInitializer {xpr, static_cast<const Scalar&>(s)};
      }
      else
      {
        auto& xpr = get_ultimate_nested_matrix(static_cast<Derived&>(*this));
        using Xpr = std::decay_t<decltype(xpr)>;
        if constexpr(mean<Derived>)
        {
          return Eigen::MeanCommaInitializer<Derived, Xpr> {xpr, static_cast<const Scalar&>(s)};
        }
        else if constexpr((Eigen3::eigen_self_adjoint_expr<Xpr> or Eigen3::eigen_triangular_expr<Xpr>)
          and diagonal_matrix<Xpr>)
        {
          return Eigen::DiagonalCommaInitializer {xpr, static_cast<const Scalar&>(s)};
        }
        else
        {
          return Eigen::CommaInitializer {xpr, static_cast<const Scalar&>(s)};
        }
      }
    }


#ifdef __cpp_concepts
    template<eigen_matrix Other>
#else
    template<typename Other, std::enable_if_t<eigen_matrix<Other>, int> = 0>
#endif
    auto operator<<(const Other& other)
    {
      if constexpr(covariance<Derived>)
      {
        auto& xpr = static_cast<Derived&>(*this);
        return Eigen::CovarianceCommaInitializer {xpr, other};
      }
      else
      {
        auto& xpr = get_ultimate_nested_matrix(static_cast<Derived&>(*this));
        using Xpr = std::decay_t<decltype(xpr)>;
        if constexpr (mean<Derived>)
        {
          return Eigen::MeanCommaInitializer<Derived, Xpr> {xpr, other};
        }
        else if constexpr ((Eigen3::eigen_self_adjoint_expr<Xpr> or Eigen3::eigen_triangular_expr<Xpr>)
          and diagonal_matrix<Xpr>)
        {
          return Eigen::DiagonalCommaInitializer {xpr, other};
        }
        else
        {
          return Eigen::CommaInitializer {xpr, other};
        }
      }
    }


  };

} // namespace OpenKalman::Eigen3::internal


namespace Eigen
{
  /**
   * \brief Alternative version of Eigen::CommaInitializer for Mean.
   */
  template<typename Derived, typename XprType>
  struct MeanCommaInitializer : CommaInitializer<XprType>
  {
    using Base = CommaInitializer<XprType>;
    using Scalar = typename OpenKalman::MatrixTraits<XprType>::Scalar;
    using Coefficients = typename OpenKalman::MatrixTraits<Derived>::RowCoefficients;
    using Base::Base;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    MeanCommaInitializer(XprType& xpr, const S& s) : Base {xpr, static_cast<const Scalar&>(s)} {}

    template<typename OtherDerived>
    MeanCommaInitializer(XprType& xpr, const DenseBase<OtherDerived>& other)
      : Base {xpr, other} {}

    ~MeanCommaInitializer()
    {
      this->m_xpr = OpenKalman::Eigen3::wrap_angles<Coefficients>(Base::finished());
    }

    auto& finished()
    {
      this->m_xpr = OpenKalman::Eigen3::wrap_angles<Coefficients>(Base::finished());
      return this->m_xpr;
    }
  };


  /**
   * \brief Version of Eigen::CommaInitializer for diagonal versions of SelfAdjointMatrix and TriangularMatrix.
   */
  template<typename XprType>
  struct DiagonalCommaInitializer
  {
    using Scalar = typename OpenKalman::MatrixTraits<XprType>::Scalar;
    static constexpr auto dim = OpenKalman::MatrixTraits<XprType>::rows;
    using NestedMatrix = OpenKalman::native_matrix_t<typename OpenKalman::nested_matrix_t<XprType>, dim, 1>;
    using Nested = CommaInitializer<NestedMatrix>;

    NestedMatrix matrix;
    Nested comma_initializer;
    XprType& diag;

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    DiagonalCommaInitializer(XprType& xpr, const S& s)
      : matrix {}, comma_initializer {matrix, static_cast<const Scalar&>(s)}, diag {xpr} {}

    template<typename OtherDerived>
    DiagonalCommaInitializer(XprType& xpr, const DenseBase<OtherDerived>& other)
      : matrix {}, comma_initializer {matrix, other}, diag {xpr} {}

    DiagonalCommaInitializer(const DiagonalCommaInitializer& o)
      : matrix {o.matrix}, comma_initializer {o.comma_initializer}, diag {o.diag} {}

    DiagonalCommaInitializer(DiagonalCommaInitializer&& o)
      : matrix {std::move(o.matrix)},
        comma_initializer {std::move(o.comma_initializer)}, diag {std::move(o.diag)} {}

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator,(const S& s)
    {
      comma_initializer, static_cast<const Scalar&>(s);
      return *this;
    }

    template<typename OtherDerived>
    auto& operator,(const DenseBase<OtherDerived>& other)
    {
      comma_initializer, other;
      return *this;
    }

    ~DiagonalCommaInitializer()
    {
      diag = OpenKalman::Eigen3::DiagonalMatrix<NestedMatrix>(comma_initializer.finished());
    }

    auto& finished()
    {
      diag = OpenKalman::Eigen3::DiagonalMatrix<NestedMatrix>(comma_initializer.finished());
      return diag;
    }
  };


  /**
   * \brief Alternative version of Eigen::CommaInitializer for Covariance and SquareRootCovariance.
   */
  template<typename CovarianceType>
  struct CovarianceCommaInitializer
  {
    using Scalar = typename OpenKalman::MatrixTraits<CovarianceType>::Scalar;
    using CovNest = typename OpenKalman::nested_matrix_t<CovarianceType>;
    using NestedMatrix = std::conditional_t<OpenKalman::diagonal_matrix<CovNest>,
      OpenKalman::native_matrix_t<CovNest, OpenKalman::MatrixTraits<CovNest>::rows, 1>,
      OpenKalman::native_matrix_t<CovNest>>;
    using Nested = CommaInitializer<NestedMatrix>;

    NestedMatrix matrix;
    Nested comma_initializer;
    CovarianceType& cov;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    CovarianceCommaInitializer(CovarianceType& xpr, const S& s)
      : matrix {}, comma_initializer {matrix, static_cast<const Scalar&>(s)}, cov {xpr} {}

    template<typename OtherDerived>
    CovarianceCommaInitializer(CovarianceType& xpr, const DenseBase<OtherDerived>& other)
      : matrix {}, comma_initializer {matrix, other}, cov {xpr} {}

    CovarianceCommaInitializer(const CovarianceCommaInitializer& o)
      : matrix {o.matrix}, comma_initializer {o.comma_initializer}, cov {o.cov} {}

    CovarianceCommaInitializer(CovarianceCommaInitializer&& o)
      : matrix {std::move(o.matrix)},
        comma_initializer {std::move(o.comma_initializer)}, cov {std::move(o.cov)} {}

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator,(const S& s)
    {
      comma_initializer, static_cast<const Scalar&>(s);
      return *this;
    }

    template<typename OtherDerived>
    auto& operator,(const DenseBase<OtherDerived>& other)
    {
      comma_initializer, other;
      return *this;
    }

    ~CovarianceCommaInitializer()
    {
      using namespace OpenKalman;

      if constexpr (diagonal_matrix<CovNest>)
      {
        cov = MatrixTraits<CovNest>::make(std::move(comma_initializer.finished()));
      }
      else if constexpr (square_root_covariance<CovarianceType>)
      {
        using T = typename MatrixTraits<CovNest>::template TriangularMatrixFrom<>;
        auto b = OpenKalman::internal::to_covariance_nestable<T>(std::move(comma_initializer.finished()));
        cov = OpenKalman::internal::to_covariance_nestable<CovNest>(std::move(b));
      }
      else
      {
        using SA = typename MatrixTraits<CovNest>::template SelfAdjointMatrixFrom<>;
        auto b = OpenKalman::internal::to_covariance_nestable<SA>(std::move(comma_initializer.finished()));
        cov = OpenKalman::internal::to_covariance_nestable<CovNest>(std::move(b));
      }
    }

    auto& finished()
    {
      using namespace OpenKalman;

      if constexpr (diagonal_matrix<CovNest>)
      {
        cov = MatrixTraits<CovNest>::make(comma_initializer.finished());
      }
      else if constexpr (square_root_covariance<CovarianceType>)
      {
        using T = typename MatrixTraits<CovNest>::template TriangularMatrixFrom<>;
        auto b = OpenKalman::internal::to_covariance_nestable<T>(comma_initializer.finished());
        cov = OpenKalman::internal::to_covariance_nestable<CovNest>(std::move(b));
      }
      else
      {
        using SA = typename MatrixTraits<CovNest>::template SelfAdjointMatrixFrom<>;
        auto b = OpenKalman::internal::to_covariance_nestable<SA>(comma_initializer.finished());
        cov = OpenKalman::internal::to_covariance_nestable<CovNest>(std::move(b));
      }
      return cov;
    }
  };

} // namespace Eigen


#endif //OPENKALMAN_EIGEN3MATRIXBASE_HPP
