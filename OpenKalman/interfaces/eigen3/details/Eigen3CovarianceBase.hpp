/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3COVARIANCEBASE_HPP
#define OPENKALMAN_EIGEN3COVARIANCEBASE_HPP

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
  /**
   * Ultimate base of OpenKalman matrix types.
   */
  template<typename Derived, typename ArgType>
  struct Eigen3CovarianceBase : Eigen3MatrixBase<Derived, ArgType>
  {
    using Nested = ArgType;
    using Scalar = typename MatrixTraits<Nested>::Scalar;
    using Base = Eigen3MatrixBase<Derived, ArgType>;

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    constexpr auto operator<<(const S& s)
    {
      if constexpr(covariance<Derived> and
        ((square_root_covariance<Derived> and not triangular_matrix<Nested>) or
          (not square_root_covariance<Derived> and not self_adjoint_matrix<Nested>)))
      {
        auto& xpr = static_cast<Derived&>(*this);
        return Eigen::CovarianceCommaInitializer(xpr, static_cast<const Scalar&>(s));
      }
      else
      {
        return Base::operator<<(s);
      }
    }


    template<typename OtherDerived>
    constexpr auto operator<<(const Eigen::DenseBase<OtherDerived>& other)
    {
      if constexpr(covariance<Derived> and
        ((square_root_covariance<Derived> and not triangular_matrix<Nested>) or
          (not square_root_covariance<Derived> and not self_adjoint_matrix<Nested>)))
      {
        auto& xpr = static_cast<Derived&>(*this);
        return Eigen::CovarianceCommaInitializer(xpr, other);
      }
      else
      {
        return Base::operator<<(other);
      }
    }
  };

} // namespace OpenKalman::Eigen3::internal


namespace Eigen
{
  /**
   * Alternative version of CommaInitializer for Mean.
   */
  template<typename Derived, typename XprType>
  struct MeanCommaInitializer : CommaInitializer<XprType>
  {
    using Base = CommaInitializer<XprType>;
    using Scalar = typename XprType::Scalar;
    using Coefficients = typename OpenKalman::MatrixTraits<Derived>::RowCoefficients;
    using Base::Base;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    MeanCommaInitializer(XprType& xpr, const S& s) : Base(xpr, static_cast<const Scalar&>(s)) {}

    template<typename OtherDerived>
    MeanCommaInitializer(XprType& xpr, const DenseBase <OtherDerived>& other) : Base(xpr, other) {}

    ~MeanCommaInitializer()
    {
      finished();
    }

    auto& finished()
    {
      this->m_xpr = OpenKalman::Eigen3::wrap_angles<Coefficients>(Base::finished());
      return this->m_xpr;
    }
  };


  /**
   * Alternative version of CommaInitializer for diagonal versions of SelfAdjointMatrix and TriangularMatrix.
   */
  template<typename XprType>
  struct DiagonalCommaInitializer
  {
    using Scalar = typename XprType::Scalar;
    static constexpr auto dim = OpenKalman::MatrixTraits<XprType>::dimension;
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
      : matrix(), comma_initializer(matrix, static_cast<const Scalar&>(s)), diag(xpr) {}

    template<typename OtherDerived>
    DiagonalCommaInitializer(XprType& xpr, const DenseBase <OtherDerived>& other)
      : matrix(), comma_initializer(matrix, other), diag(xpr) {}

    DiagonalCommaInitializer(const DiagonalCommaInitializer& o)
      : matrix(o.matrix), comma_initializer(o.comma_initializer), diag(o.diag) {}

    DiagonalCommaInitializer(DiagonalCommaInitializer&& o)
      : matrix(std::move(o.matrix)),
        comma_initializer(std::move(o.comma_initializer)), diag(std::move(o.diag)) {}

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
      finished();
    }

    auto& finished()
    {
      diag = OpenKalman::Eigen3::DiagonalMatrix<NestedMatrix>(comma_initializer.finished());
      return diag;
    }
  };


  /**
   * Alternative version of CommaInitializer for Covariance and SquareRootCovariance.
   */
  template<typename CovarianceType>
  struct CovarianceCommaInitializer
  {
    using Scalar = typename CovarianceType::Scalar;
    using NestedMatrix = OpenKalman::native_matrix_t<typename OpenKalman::nested_matrix_t<CovarianceType>>;
    using Nested = CommaInitializer<NestedMatrix>;

    NestedMatrix matrix;
    Nested comma_initializer;
    CovarianceType& cov;

    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    CovarianceCommaInitializer(CovarianceType& xpr, const S& s)
      : matrix(), comma_initializer(matrix, static_cast<const Scalar&>(s)), cov(xpr) {}

    template<typename OtherDerived>
    CovarianceCommaInitializer(CovarianceType& xpr, const DenseBase <OtherDerived>& other)
      : matrix(), comma_initializer(matrix, other), cov(xpr) {}

    CovarianceCommaInitializer(const CovarianceCommaInitializer& o)
      : matrix(o.matrix), comma_initializer(o.comma_initializer), cov(o.cov) {}

    CovarianceCommaInitializer(CovarianceCommaInitializer&& o)
      : matrix(std::move(o.matrix)),
        comma_initializer(std::move(o.comma_initializer)), cov(std::move(o.cov)) {}

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
      finished();
    }

    auto& finished()
    {
      cov = comma_initializer.finished();
      return cov;
    }
  };

} // namespace Eigen


#endif //OPENKALMAN_EIGEN3COVARIANCEBASE_HPP
