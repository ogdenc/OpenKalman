/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COVARIANCEBASE_H
#define OPENKALMAN_COVARIANCEBASE_H

#include <utility>

namespace OpenKalman::internal
{

  /**
   * \internal
   * \brief Covariance Cov's cholesky nested matrix and nested matrix Nested are both either triangular or self-adjoint.
   */
  template<typename Cov, typename Nested = nested_matrix_t<Cov>>
#ifdef __cpp_concepts
  concept case1or2 = (not square_root_covariance<Cov> and self_adjoint_matrix<Nested>) or
    (square_root_covariance<Cov> and triangular_matrix<Nested>);
#else
  static constexpr bool case1or2 = (not square_root_covariance<Cov> and self_adjoint_matrix<Nested>) or
    (square_root_covariance<Cov> and triangular_matrix<Nested>);
#endif


  // =====================================CASE 1=======================================
  /**
   * \internal
   * \anchor CovarianceBaseCase1
   * \brief Base of Covariance and SquareRootCovariance classes (Case 1).
   * \details This specialization is operative if NestedMatrix is self-contained and either
   * # Derived is not a square root and NestedMatrix is self-adjoint; or
   * # Derived is a square root and NestedMatrix is triangular.
   * In this case, NestedMatrix and the cholesky nested matrix are the same.
   */
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix> requires
    case1or2<Derived, NestedMatrix> and self_contained<NestedMatrix>
  struct CovarianceBase<Derived, NestedMatrix>
#else
  template<typename Derived, typename NestedMatrix>
  struct CovarianceBase<Derived, NestedMatrix, std::enable_if_t<
    case1or2<Derived, NestedMatrix> and self_contained<NestedMatrix>>>
#endif
    : MatrixBase<Derived, NestedMatrix>
  {
  private:

    using Base = MatrixBase<Derived, NestedMatrix>;

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;

  protected:

    using CholeskyNestedMatrix = std::conditional_t<triangular_matrix<NestedMatrix>,
      typename MatrixTraits<NestedMatrix>::template SelfAdjointMatrixFrom<>,
      typename MatrixTraits<NestedMatrix>::template TriangularMatrixFrom<>>;

    /**
     * \internal
     * \brief The cholesky factor or square of the nested matrix.
     * \details This will be triangular if Covariance or self-adjoint if SquareRootCovariance
     */
    CholeskyNestedMatrix cholesky_nested_matrix() const
    {
      if constexpr (square_root_covariance<Derived>) return Cholesky_square(Base::nested_matrix());
      else return Cholesky_factor(Base::nested_matrix());
    }


    /**
     * \internal
     * \brief The synchronization direction. 0 == synchronized, 1 = forward, -1 = reverse.
     */
    constexpr static auto synchronization_direction = []() -> int { return 0; };


    /**
     * \internal
     * \brief Synchronize the state from nested_matrix to cholesky_nested_matrix.
     * \details This is a no-op.
     */
    constexpr static auto synchronize_forward = [] {};


    /**
     * \internal
     * \brief Synchronize the state from cholesky_nested_matrix to nested_matrix.
     * \details This is a no-op.
     */
    constexpr static auto synchronize_reverse = [] {};


    /**
     * \internal
     * \brief Indicate that the nested matrix has changed.
     * \details In this specialization of CovarianceBase, this function is a no-op.
     */
    constexpr static auto mark_nested_matrix_changed = [] {};


    /**
     * \internal
     * \brief Indicate that the cholesky nested matrix has changed.
     * \details In this specialization of CovarianceBase, this function is a no-op.
     */
    constexpr static auto mark_cholesky_nested_matrix_changed = [] {};


    /**
     * \internal
     * \brief Indicate that the covariance is synchronized.
     * \details In this specialization of CovarianceBase, this function is a no-op.
     */
    constexpr static auto mark_synchronized = [] {};


  public:
    /// Default constructor.
#ifdef __cpp_concepts
    CovarianceBase() requires std::default_initializable<Base>
#else
    template<typename T = Base, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    CovarianceBase()
#endif
      : Base {} {}


    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other) : Base {other} {}


    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept : Base {std::move(other)} {}


    /**
     * \brief Construct from another \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not std::derived_from<std::decay_t<Arg>, CovarianceBase>) and
      std::is_constructible_v<Base, decltype(to_covariance_nestable<NestedMatrix>(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>) and
      std::is_constructible_v<Base, NestedMatrix&&>, int> = 0>
        // std::is_constructible_v cannot be used here with to_covariance_nestable.
#endif
    CovarianceBase(Arg&& arg) noexcept : Base {to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg))} {}


    /**
     * \brief Construct from a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg> requires
      std::is_constructible_v<Base, decltype(to_covariance_nestable<NestedMatrix>(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg> and
      std::is_constructible_v<Base, NestedMatrix&&>, int> = 0>
        // std::is_constructible_v cannot be used here with to_covariance_nestable.
#endif
    explicit CovarianceBase(Arg&& arg) noexcept : Base {to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg))} {}


    /**
     * \internal
     * \brief Copy assignment operator.
     */
    auto& operator=(const CovarianceBase& other)
    {
      Base::operator=(other);
      return *this;
    }


    /**
     * \internal
     * \brief Move assignment operator.
     */
    auto& operator=(CovarianceBase&& other) noexcept
    {
      Base::operator=(std::move(other));
      return *this;
    }


    /**
     * \internal
     * \brief Assign from another \ref covariance or a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (covariance<Arg> or covariance_nestable<Arg>) and
      (not std::derived_from<std::decay_t<Arg>, CovarianceBase>)
#else
    template<typename Arg, std::enable_if_t<(covariance<Arg> or covariance_nestable<Arg>) and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>), int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      Base::operator=(to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg)));
      return *this;
    }


    /**
     * \brief Get or set element (i, j) of the covariance matrix.
     * \param i The row.
     * \param j The column.
     * \return An ElementSetter object.
     */
    auto operator() (std::size_t i, std::size_t j)
    {
      return make_ElementSetter<not element_settable<NestedMatrix, 2>>(Base::nested_matrix(), i, j);
    }

    /// \overload
    auto operator() (std::size_t i, std::size_t j) const
    {
      return make_ElementSetter<true>(Base::nested_matrix(), i, j);
    }


    /**
     * \brief Get or set element i of the covariance matrix, if it is a vector.
     * \param i The row.
     * \param j The column.
     * \return An ElementSetter object.
     */
    auto operator[] (std::size_t i)
    {
      return make_ElementSetter<not element_settable<NestedMatrix, 1>>(Base::nested_matrix(), i);
    }


    /// \overload
    auto operator[] (std::size_t i) const
    {
      return make_ElementSetter<true>(Base::nested_matrix(), i);
    }

    /// \overload
    auto operator() (std::size_t i) { return operator[](i); }

    /// \overload
    auto operator() (std::size_t i) const { return operator[](i); }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i, const std::size_t j)
    {
      OpenKalman::set_element(Base::nested_matrix(), s, i, j);
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i)
    {
      OpenKalman::set_element(Base::nested_matrix(), s, i);
    }

  };


  // ======================================CASE 2======================================
  /**
   * \internal
   * \anchor CovarianceBaseCase2
   * \brief Base of Covariance and SquareRootCovariance classes (Case 2).
   * \details This specialization is operative if NestedMatrix is not self-contained and either
   * # Derived is not a square root and NestedMatrix is self-adjoint; or
   * # Derived is a square root and NestedMatrix is triangular.
   * In this case, NestedMatrix and the cholesky nested matrix are the same.
   */
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix> requires
    case1or2<Derived, NestedMatrix> and (not self_contained<NestedMatrix>)
  struct CovarianceBase<Derived, NestedMatrix>
#else
  template<typename Derived, typename NestedMatrix>
  struct CovarianceBase<Derived, NestedMatrix, std::enable_if_t<
    case1or2<Derived, NestedMatrix> and (not self_contained<NestedMatrix>)>>
#endif
    : MatrixBase<Derived, NestedMatrix>
  {
  private:

    using Base = MatrixBase<Derived, NestedMatrix>;

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;

  protected:

    using CholeskyNestedMatrix = std::conditional_t<triangular_matrix<NestedMatrix>,
      typename MatrixTraits<NestedMatrix>::template SelfAdjointMatrixFrom<>,
      typename MatrixTraits<NestedMatrix>::template TriangularMatrixFrom<>>;

  private:

    // The Cholesky square or square root of the nested matrix.
    const std::function<CholeskyNestedMatrix()> cholesky_nested;

  protected:

    /**
     * \internal
     * \brief The Cholesky nested matrix: triangular if Covariance, self-adjoint if SquareRootCovariance
     */
    CholeskyNestedMatrix cholesky_nested_matrix() const { return cholesky_nested(); }


    /**
     * \internal
     * \brief The synchronization direction. 0 == synchronized, 1 = forward, -1 = reverse.
     */
    const std::function<int()> synchronization_direction;


    /**
     * \internal
     * \brief Synchronize the state from nested_matrix to cholesky_nested_matrix.
     * \details This is a no-op.
     */
    const std::function<void()> synchronize_forward;


    /**
     * \internal
     * \brief Synchronize the state from cholesky_nested_matrix to nested_matrix.
     * \details This is a no-op.
     */
    const std::function<void()> synchronize_reverse;


    /**
     * \internal
     * \brief Indicate that the nested matrix has changed.
     */
    const std::function<void()> mark_nested_matrix_changed;


    /**
     * \internal
     * \brief Indicate that the cholesky nested matrix has changed.
     * \details In this specialization of CovarianceBase, this function is a no-op.
     */
    constexpr static auto mark_cholesky_nested_matrix_changed = [] {};


    /**
     * \internal
     * \brief Indicate that the covariance is synchronized.
     * \details In this specialization of CovarianceBase, this function is a no-op.
     */
    constexpr static auto mark_synchronized = [] {};


  public:
    /// Default constructor.
    CovarianceBase() = delete;


    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other)
      : Base {other},
        cholesky_nested {other.cholesky_nested},
        synchronization_direction {other.synchronization_direction},
        synchronize_forward {other.synchronize_forward},
        synchronize_reverse {other.synchronize_reverse},
        mark_nested_matrix_changed {other.mark_nested_matrix_changed}
        {}


    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : Base {std::move(other)},
        cholesky_nested {std::move(other.cholesky_nested)},
        synchronization_direction {std::move(other.synchronization_direction)},
        synchronize_forward {std::move(other.synchronize_forward)},
        synchronize_reverse {std::move(other.synchronize_reverse)},
        mark_nested_matrix_changed {std::move(other.mark_nested_matrix_changed)}
        {}


    /**
     * \brief Construct from another \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not std::derived_from<std::decay_t<Arg>, CovarianceBase>) and
      std::is_constructible_v<Base, decltype(std::declval<Arg>().nested_matrix())>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>) and
      std::is_constructible_v<Base, decltype(std::declval<Arg>().nested_matrix())>, int> = 0>
#endif
    CovarianceBase(Arg&& arg)
      : Base {std::forward<decltype(arg.nested_matrix())>(arg.nested_matrix())},
        cholesky_nested {[f = [&] { return arg.cholesky_nested_matrix(); }] { return f(); }},
        synchronization_direction {arg.synchronization_direction},
        synchronize_forward {arg.synchronize_forward},
        synchronize_reverse {arg.synchronize_reverse},
        mark_nested_matrix_changed {arg.mark_nested_matrix_changed}
        {}


    /**
     * \brief Construct from a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg> requires std::is_constructible_v<Base, Arg> and std::is_constructible_v<Base, Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg> and std::is_constructible_v<Base, Arg> and
      std::is_constructible_v<Base, Arg>, int> = 0>
#endif
    explicit CovarianceBase(Arg&& arg) noexcept
      : Base {std::forward<Arg>(arg)},
        cholesky_nested {[&n = Base::nested_matrix()] {
          if constexpr (square_root_covariance<Derived>) return Cholesky_square(n);
          else return Cholesky_factor(n);
        }},
        synchronization_direction {[] { return 0; }},
        synchronize_forward {[] {}},
        synchronize_reverse {[] {}},
        mark_nested_matrix_changed {[] {}}
        {}


    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        Base::operator=(other);
        mark_nested_matrix_changed();
      }
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        Base::operator=(std::move(other));
        mark_nested_matrix_changed();
      }
      return *this;
    }


    /**
     * \internal
     * \brief Assign from another \ref covariance or a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (covariance<Arg> or covariance_nestable<Arg>) and
      (not std::derived_from<std::decay_t<Arg>, CovarianceBase>)
#else
    template<typename Arg, std::enable_if_t<(covariance<Arg> or covariance_nestable<Arg>) and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>), int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      Base::operator=(to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg)));
      mark_nested_matrix_changed();
      return *this;
    }


    /**
     * \brief Get or set element (i, j) of the covariance matrix.
     * \param i The row.
     * \param j The column.
     * \return An ElementSetter object.
     */
    auto operator() (std::size_t i, std::size_t j)
    {
      if constexpr (element_settable<NestedMatrix, 2>)
        return ElementSetter(Base::nested_matrix(), i, j,
          [this] { if (synchronization_direction() < 0) synchronize_reverse(); },
          [this] { mark_nested_matrix_changed(); });
      else
        return make_ElementSetter<true>(Base::nested_matrix(), i, j);
    }


    /// \overload
    auto operator() (std::size_t i, std::size_t j) const
    {
      return make_ElementSetter<true>(Base::nested_matrix(), i, j);
    }


    /**
     * \brief Get or set element i of the covariance matrix, if it is a vector.
     * \param i The row.
     * \param j The column.
     * \return An ElementSetter object.
     */
    auto operator[] (std::size_t i)
    {
      if constexpr (element_settable<NestedMatrix, 1>)
        return ElementSetter(Base::nested_matrix(), i,
          [this] { if (synchronization_direction() < 0) synchronize_reverse(); },
          [this] { mark_nested_matrix_changed(); });
      else
        return make_ElementSetter<true>(Base::nested_matrix(), i);
    }

    /// \overload
    auto operator[] (std::size_t i) const
    {
      return make_ElementSetter<true>(Base::nested_matrix(), i);
    }

    /// \overload
    auto operator() (std::size_t i) { return operator[](i); }

    /// \overload
    auto operator() (std::size_t i) const { return operator[](i); }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i, const std::size_t j)
    {
      OpenKalman::set_element(Base::nested_matrix(), s, i, j);
      mark_nested_matrix_changed();
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i)
    {
      OpenKalman::set_element(Base::nested_matrix(), s, i);
      mark_nested_matrix_changed();
    }

  };


  // ======================================CASE 3======================================
  /**
   * \internal
   * \anchor CovarianceBaseCase3
   * \brief Base of Covariance and SquareRootCovariance classes (Case 3).
   * \details This specialization is operative if ArgType is self-contained and
   * # Derived is a square root or NestedMatrix is not self-adjoint; and
   * # Derived is not a square root or NestedMatrix is not triangular.
   * In this case, NestedMatrix and the cholesky nested matrix are different.
   */
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix> requires
    (not case1or2<Derived, NestedMatrix>) and self_contained<NestedMatrix>
  struct CovarianceBase<Derived, NestedMatrix>
#else
  template<typename Derived, typename NestedMatrix>
  struct CovarianceBase<Derived, NestedMatrix, std::enable_if_t<
    (not case1or2<Derived, NestedMatrix>) and self_contained<NestedMatrix>>>
#endif
    : MatrixBase<Derived, NestedMatrix>
  {
  private:

    using Base = MatrixBase<Derived, NestedMatrix>;

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;

  protected:

    using CholeskyNestedMatrix = std::conditional_t<triangular_matrix<NestedMatrix>,
      typename MatrixTraits<NestedMatrix>::template SelfAdjointMatrixFrom<>,
      typename MatrixTraits<NestedMatrix>::template TriangularMatrixFrom<>>;

  private:

    // The cholesky nested matrix one would expect given whether the covariance is a square root.
    mutable CholeskyNestedMatrix cholesky_nested;

    // The synchronization state. 0 == synchronized, 1 = forward, -1 = reverse.
    mutable int synch_direction;

    using MutableNestedMatrix = std::conditional_t<std::is_reference_v<NestedMatrix>, NestedMatrix,
      std::remove_const_t<NestedMatrix>>;

    // Get a mutable version of the nested matrix.
    auto& mutable_nested_matrix() const &
    {
      return const_cast<std::add_lvalue_reference_t<MutableNestedMatrix>>(Base::nested_matrix());
    }

    auto&& mutable_nested_matrix() const &&
    {
      return const_cast<std::add_rvalue_reference_t<MutableNestedMatrix>>(std::move(Base::nested_matrix()));
    }

    static constexpr bool nested_is_modifiable = modifiable<MutableNestedMatrix, NestedMatrix> and
      std::is_default_constructible_v<MutableNestedMatrix>;

#ifdef __cpp_concepts
    template<typename, typename>
    friend struct internal::CovarianceBase;
#else
    template<typename, typename, typename>
    friend struct internal::CovarianceBase;
#endif

  protected:

    /**
     * \internal
     * \brief The cholesky nested matrix: self-adjoint if Covariance, triangular if SquareRootCovariance
     */
    CholeskyNestedMatrix& cholesky_nested_matrix() & { return cholesky_nested; }

    /// \overload
    CholeskyNestedMatrix&& cholesky_nested_matrix() && { return std::move(cholesky_nested); }

    /// \overload
    const CholeskyNestedMatrix& cholesky_nested_matrix() const & { return cholesky_nested; }

    /// \overload
    const CholeskyNestedMatrix&& cholesky_nested_matrix() const && { return std::move(cholesky_nested); }


    /**
     * \internal
     * \brief The synchronization direction. 0 == synchronized, 1 = forward, -1 = reverse.
     */
    const std::function<int()> synchronization_direction = [this] { return synch_direction; };


    /**
     * \internal
     * \brief Synchronize from nested_matrix to cholesky_nested_matrix.
     */
    const std::function<void()> synchronize_forward = [this] {
      cholesky_nested = to_covariance_nestable<CholeskyNestedMatrix>(Base::nested_matrix());
      synch_direction = 0;
    };


    /**
     * \internal
     * \brief Synchronize from cholesky_nested_matrix to nested_matrix.
     */
    const std::function<void()> synchronize_reverse = [this] {
      if constexpr (nested_is_modifiable)
      {
        mutable_nested_matrix() = to_covariance_nestable<NestedMatrix>(cholesky_nested);
      }
      else throw std::logic_error {"Case 3 synchronize_reverse: NestedMatrix is not modifiable."};
      synch_direction = 0;
    };


    /**
     * \internal
     * \brief Indicate that the nested matrix has changed.
     */
    const std::function<void()> mark_nested_matrix_changed = [this] { synch_direction = 1; };


    /**
     * \internal
     * \brief Indicate that the cholesky nested matrix has changed.
     */
    const std::function<void()> mark_cholesky_nested_matrix_changed = [this] { synch_direction = -1; };


    /**
     * \internal
     * \brief Indicate that the covariance is synchronized.
     */
    const std::function<void()> mark_synchronized = [this] { synch_direction = 0; };


  public:
    /// Default constructor.
#ifdef __cpp_concepts
    CovarianceBase() requires std::default_initializable<Base>
#else
    template<typename T = Base, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    CovarianceBase()
#endif
      : Base {}, synch_direction {} {}


    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other)
      : Base {other},
        cholesky_nested {other.cholesky_nested},
        synch_direction {other.synch_direction} {}


    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : Base {std::move(other)},
        cholesky_nested {std::move(other.cholesky_nested)},
        synch_direction {other.synch_direction} {}


    /**
     * \brief Construct from another \ref covariance.
     * \details This overload is operative if Arg's nested matrix and its cholesky nested matrix are the same.
     * In other words, Arg is a \ref CovarianceBaseCase3 "Case 3" or \ref CovarianceBaseCase4 "Case 4" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires
      (not std::derived_from<std::decay_t<Arg>, CovarianceBase>) and (not case1or2<Arg>) and
      std::is_constructible_v<Base, decltype(to_covariance_nestable<NestedMatrix>(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>) and (not case1or2<Arg>) and
      std::is_constructible_v<Base, NestedMatrix&&>, int> = 0>
        // std::is_constructible_v cannot be used here with to_covariance_nestable.
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base {to_covariance_nestable<NestedMatrix>(arg)},
        cholesky_nested {to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg))},
        synch_direction {0} {}


    /**
     * \brief Construct from another \ref covariance.
     * \details This overload is operative if Arg's nested matrix is the same as its cholesky nested matrix
     * as well as NestedMatrix. This only occurs when constructing the square or square_root of a matrix.
     * In other words, Arg is a \ref CovarianceBaseCase1 "Case 1" or \ref CovarianceBaseCase2 "Case 2" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires case1or2<Arg> and
      (triangular_matrix<nested_matrix_t<Arg>> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>) and
      std::is_constructible_v<Base, decltype(to_covariance_nestable<NestedMatrix>(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and case1or2<Arg> and
      (triangular_matrix<nested_matrix_t<Arg>> == triangular_matrix<NestedMatrix>) and
      (not diagonal_matrix<Arg>) and
      std::is_constructible_v<Base, NestedMatrix&&>, int> = 0>
        // std::is_constructible_v cannot be used here with to_covariance_nestable.
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base {to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg))},
        cholesky_nested {},
        synch_direction {1} {}


    /**
     * \brief Construct from another \ref covariance.
     * \details This overload is operative if Arg's nested matrix is the same as its cholesky nested matrix
     * but not NestedMatrix.
     * In other words, Arg is a \ref CovarianceBaseCase1 "Case 1" or \ref CovarianceBaseCase2 "Case 2" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires case1or2<Arg> and
      (triangular_matrix<nested_matrix_t<Arg>> != triangular_matrix<NestedMatrix> or diagonal_matrix<Arg>) and
      std::is_constructible_v<Base, NestedMatrix&&>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and case1or2<Arg> and
      (triangular_matrix<nested_matrix_t<Arg>> != triangular_matrix<NestedMatrix> or diagonal_matrix<Arg>) and
      std::is_constructible_v<Base, NestedMatrix&&>, int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base {[&] {
          if constexpr (nested_is_modifiable) return NestedMatrix {};
          else return NestedMatrix {to_covariance_nestable<NestedMatrix>(arg)};
          }()},
        cholesky_nested {to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg))},
        synch_direction {nested_is_modifiable ? -1 : 0} {}


    /**
     * \brief Construct from a \ref covariance_nestable of matching triangle/self-adjoint type.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg> requires
      (triangular_matrix<Arg> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>) and
      std::is_constructible_v<Base, decltype(to_covariance_nestable<NestedMatrix>(std::declval<Arg>()))>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg> and
      (triangular_matrix<Arg> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>) and
      std::is_constructible_v<Base, NestedMatrix&&>, int> = 0>
        // std::is_constructible_v cannot be used here with to_covariance_nestable.
#endif
    explicit CovarianceBase(Arg&& arg) noexcept
      : Base {to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg))},
        cholesky_nested {},
        synch_direction {1} {}


    /**
     * \brief Construct from a \ref covariance_nestable of non-matching triangle/self-adjoint type.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg> requires
      (triangular_matrix<Arg> != triangular_matrix<NestedMatrix> or diagonal_matrix<Arg>) and
      std::is_constructible_v<Base, NestedMatrix&&>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg> and
      (triangular_matrix<Arg> != triangular_matrix<NestedMatrix> or diagonal_matrix<Arg>) and
      std::is_constructible_v<Base, NestedMatrix&&>, int> = 0>
#endif
    explicit CovarianceBase(Arg&& arg) noexcept
      : Base {[&] {
          if constexpr (nested_is_modifiable) return NestedMatrix {};
          else return NestedMatrix {to_covariance_nestable<NestedMatrix>(arg)};
        }()},
        cholesky_nested {to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg))},
        synch_direction {nested_is_modifiable ? -1 : 0} {}


    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        if (synch_direction >= 0) Base::nested_matrix() = other.nested_matrix();
        if (synch_direction <= 0) cholesky_nested = other.cholesky_nested_matrix();
        synch_direction = other.synch_direction;
      }
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        if (synch_direction >= 0) Base::nested_matrix() = std::move(other.nested_matrix());
        if (synch_direction <= 0) cholesky_nested = std::move(other.cholesky_nested_matrix());
        synch_direction = other.synch_direction;
      }
      return *this;
    }


    /**
     * \internal
     * \brief Assign from another \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not std::derived_from<std::decay_t<Arg>, CovarianceBase>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>), int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr(not (zero_matrix<nested_matrix_t<Arg>> and zero_matrix<NestedMatrix>) and
        not (identity_matrix<nested_matrix_t<Arg>> and identity_matrix<NestedMatrix>))
      {
        if constexpr (case1or2<Arg>)
        {
          // Arg is Case 1 or 2
          if constexpr (triangular_matrix<nested_matrix_t<Arg>> == triangular_matrix<NestedMatrix> and
            not diagonal_matrix<Arg>)
          {
            Base::nested_matrix() = to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg));
            mark_nested_matrix_changed();
          }
          else
          {
            cholesky_nested = to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg));
            mark_cholesky_nested_matrix_changed();
          }
        }
        else
        {
          // Arg is Case 3 or 4
          if (synch_direction >= 0)
          {
            Base::nested_matrix() = to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg));
          }
          if (synch_direction <= 0)
          {
            cholesky_nested = to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg));
          }
          synch_direction = arg.synchronization_direction();
        }
      }
      return *this;
    }


    /**
     * \brief Assign from a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr(not (zero_matrix<Arg> and zero_matrix<NestedMatrix>) and
        not (identity_matrix<Arg> and identity_matrix<NestedMatrix>))
      {
        if constexpr(triangular_matrix<Arg> == triangular_matrix<NestedMatrix> and not diagonal_matrix<Arg>)
        {
          Base::nested_matrix() = to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg));
          mark_nested_matrix_changed();
        }
        else
        {
          cholesky_nested = to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg));
          mark_cholesky_nested_matrix_changed();
        }
      }
      return *this;
    }


    /**
     * \brief Get or set element (i, j) of the covariance matrix.
     * \param i The row.
     * \param j The column.
     * \return An ElementSetter object.
     */
    auto operator() (std::size_t i, std::size_t j)
    {
      if constexpr(element_settable<CholeskyNestedMatrix, 2>)
        return ElementSetter(cholesky_nested, i, j,
          [this] { if (synch_direction > 0) synchronize_forward(); },
          [this] { mark_cholesky_nested_matrix_changed(); });
      else
        return make_ElementSetter<true>(cholesky_nested_matrix(), i, j,
          [this] { if (synch_direction > 0) synchronize_forward(); });
    }

    /// \overload
    auto operator() (std::size_t i, std::size_t j) const
    {
      return make_ElementSetter<true>(cholesky_nested, i, j,
        [this] { if (synch_direction > 0) synchronize_forward(); });
    }


    /**
     * \brief Get or set element i of the covariance matrix.
     * \param i The row.
     * \return An ElementSetter object.
     */
    auto operator[] (std::size_t i)
    {
      if constexpr(element_settable<CholeskyNestedMatrix, 1>)
        return ElementSetter(cholesky_nested, i,
          [this] { if (synch_direction > 0) synchronize_forward(); },
          [this] { mark_cholesky_nested_matrix_changed(); });
      else
        return make_ElementSetter<true>(cholesky_nested, i,
          [this] { if (synch_direction > 0) synchronize_forward(); });
    }

    /// \overload
    auto operator[] (std::size_t i) const
    {
      return make_ElementSetter<true>(cholesky_nested, i,
        [this] { if (synch_direction > 0) synchronize_forward(); });
    }

    /// \overload
    auto operator() (std::size_t i) { return operator[](i); }

    /// \overload
    auto operator() (std::size_t i) const { return operator[](i); }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i, const std::size_t j)
    {
      if (synch_direction > 0) synchronize_forward();
      OpenKalman::set_element(cholesky_nested, s, i, j);
      mark_cholesky_nested_matrix_changed();
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i)
    {
      if (synch_direction > 0) synchronize_forward();
      OpenKalman::set_element(cholesky_nested, s, i);
      mark_cholesky_nested_matrix_changed();
    }

  };


  // ======================================CASE 4======================================
  /*
   * \internal
   * \anchor CovarianceBaseCase4
   * \brief Base of Covariance and SquareRootCovariance classes (Case 4).
   * \details This specialization is operative if NestedMatrix is not self-contained and
   * # Derived is a square root or NestedMatrix is not self-adjoint; and
   * # Derived is not a square root or NestedMatrix is not triangular.
   * In this case, NestedMatrix and the cholesky nested matrix are different.
   */
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix> requires
    (not case1or2<Derived, NestedMatrix>) and (not self_contained<NestedMatrix>)
  struct CovarianceBase<Derived, NestedMatrix>
#else
  template<typename Derived, typename NestedMatrix>
  struct CovarianceBase<Derived, NestedMatrix, std::enable_if_t<
    (not case1or2<Derived, NestedMatrix>) and (not self_contained<NestedMatrix>)>>
#endif
  : MatrixBase<Derived, NestedMatrix>
  {
  private:

    using Base = MatrixBase<Derived, NestedMatrix>;

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;

  protected:

    using CholeskyNestedMatrix = std::conditional_t<triangular_matrix<NestedMatrix>,
      typename MatrixTraits<NestedMatrix>::template SelfAdjointMatrixFrom<>,
      typename MatrixTraits<NestedMatrix>::template TriangularMatrixFrom<>>;

  private:

    // The cholesky nested matrix, which is self-adjoint if Covariance, triangular if SquareRootCovariance.
    const std::function<CholeskyNestedMatrix&()> cholesky_nested;

    using MutableNestedMatrix = std::conditional_t<std::is_reference_v<NestedMatrix>, NestedMatrix,
      std::remove_const_t<NestedMatrix>>;

    // Get a mutable version of the nested matrix.
    constexpr auto& mutable_nested_matrix() const &
    {
      return const_cast<std::add_lvalue_reference_t<MutableNestedMatrix>>(Base::nested_matrix());
    }

    constexpr auto&& mutable_nested_matrix() const &&
    {
      return const_cast<std::add_rvalue_reference_t<MutableNestedMatrix>>(std::move(Base::nested_matrix()));
    }

    static constexpr bool nested_is_modifiable = modifiable<MutableNestedMatrix, NestedMatrix>;

#ifdef __cpp_concepts
    template<typename, typename>
    friend struct internal::CovarianceBase;
#else
    template<typename, typename, typename>
    friend struct internal::CovarianceBase;
#endif

  protected:

    /**
     * \internal
     * \brief The cholesky nested matrix: self-adjoint if Covariance, triangular if SquareRootCovariance
     */
    CholeskyNestedMatrix& cholesky_nested_matrix() & { return cholesky_nested(); }

    /// \overload
    CholeskyNestedMatrix& cholesky_nested_matrix() && { return cholesky_nested(); }

    /// \overload
    const CholeskyNestedMatrix& cholesky_nested_matrix() const & { return cholesky_nested(); }

    /// \overload
    const CholeskyNestedMatrix& cholesky_nested_matrix() const && { return cholesky_nested(); }


    /**
     * \internal
     * \brief The synchronization direction. 0 == synchronized, 1 = forward, -1 = reverse.
     */
    const std::function<int()> synchronization_direction;


    /**
     * \internal
     * \brief Synchronize nested_matrix to cholesky_nested_matrix.
     */
    const std::function<void()> synchronize_forward;


    /**
     * \internal
     * \brief Synchronize cholesky_nested_matrix to nested_matrix.
     */
    const std::function<void()> synchronize_reverse;


    /**
     * \internal
     * \brief Indicate that the nested matrix has changed.
     */
    const std::function<void()> mark_nested_matrix_changed;


    /**
     * \internal
     * \brief Indicate that the cholesky nested matrix has changed.
     */
    const std::function<void()> mark_cholesky_nested_matrix_changed;


    /**
     * \internal
     * \brief Indicate that the covariance is synchronized.
     */
    const std::function<void()> mark_synchronized;


  public:
    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other)
      : Base {other},
        cholesky_nested {[&ch = other.cholesky_nested()]() -> CholeskyNestedMatrix& { return ch; }},
        synchronization_direction {other.synchronization_direction},
        synchronize_forward {other.synchronize_forward},
        synchronize_reverse {other.synchronize_reverse},
        mark_nested_matrix_changed {other.mark_nested_matrix_changed},
        mark_cholesky_nested_matrix_changed {other.mark_cholesky_nested_matrix_changed},
        mark_synchronized {other.mark_synchronized}
    {}


    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : Base {std::move(other)},
        cholesky_nested {std::move(other.cholesky_nested)},
        synchronization_direction {std::move(other.synchronization_direction)},
        synchronize_forward {std::move(other.synchronize_forward)},
        synchronize_reverse {std::move(other.synchronize_reverse)},
        mark_nested_matrix_changed {std::move(other.mark_nested_matrix_changed)},
        mark_cholesky_nested_matrix_changed {std::move(other.mark_cholesky_nested_matrix_changed)},
        mark_synchronized {std::move(other.mark_synchronized)}
    {}


    /**
     * \brief Construct from another \ref covariance.
     * \details This overload is operative if Arg's nested matrix is different from its cholesky nested matrix.
     * In other words, Arg is a \ref CovarianceBaseCase3 "Case 3" or \ref CovarianceBaseCase4 "Case 4" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not case1or2<Arg>) and
      (not std::derived_from<std::decay_t<Arg>, CovarianceBase>) and
      std::is_constructible_v<Base, decltype(std::declval<Arg>().nested_matrix())>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not case1or2<Arg>) and (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>) and
      std::is_constructible_v<Base, decltype(std::declval<Arg>().nested_matrix())>, int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base {std::forward<decltype(arg.nested_matrix())>(arg.nested_matrix())},
        cholesky_nested {[&ch = arg.cholesky_nested_matrix()]() -> CholeskyNestedMatrix& { return ch; }},
        synchronization_direction {arg.synchronization_direction},
        synchronize_forward {arg.synchronize_forward},
        synchronize_reverse {arg.synchronize_reverse},
        mark_nested_matrix_changed {arg.mark_nested_matrix_changed},
        mark_cholesky_nested_matrix_changed {arg.mark_cholesky_nested_matrix_changed},
        mark_synchronized {arg.mark_synchronized}
    {}


    /**
     * \brief Construct from another \ref covariance.
     * \details This overload is operative if Arg's nested matrix is the same kind as its cholesky nested matrix
     * and the same as NestedMatrix. This only occurs when constructing the square or square_root of a covariance.
     * In other words, Arg is a \ref CovarianceBaseCase1 "Case 1" or \ref CovarianceBaseCase2 "Case 2" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires case1or2<Arg> and
      (triangular_matrix<nested_matrix_t<Arg>> == triangular_matrix<NestedMatrix>) and
      (not diagonal_matrix<Arg>) and std::is_constructible_v<Base, decltype(std::declval<Arg>().nested_matrix())>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and case1or2<Arg> and
      (triangular_matrix<nested_matrix_t<Arg>> == triangular_matrix<NestedMatrix>) and
      (not diagonal_matrix<Arg>) and std::is_constructible_v<Base, decltype(std::declval<Arg>().nested_matrix())>,
        int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base {std::forward<decltype(arg.nested_matrix())>(arg.nested_matrix())},
        cholesky_nested {[a = CholeskyNestedMatrix {}]() mutable -> CholeskyNestedMatrix& { return a; }},
        synchronization_direction {[] { return 1; }},
        synchronize_forward {
          [&ch = cholesky_nested(), s = arg.synchronization_direction,
            f = arg.synchronize_forward, c = [&] { return arg.cholesky_nested_matrix(); }] {
          if (s() > 0) f();
          ch = c();
        }},
        synchronize_reverse {[&ch = cholesky_nested(), &m = mutable_nested_matrix()] {
          if constexpr (nested_is_modifiable) m = to_covariance_nestable<NestedMatrix>(ch);
          else throw std::logic_error {"Case 4 synchronize_reverse: NestedMatrix is not modifiable."};
        }},
        mark_nested_matrix_changed {[] {}},
        mark_cholesky_nested_matrix_changed {[] {}},
        mark_synchronized {[] {}}
    {}


    /**
     * \brief Construct from a \ref covariance_nestable of matching triangle/self-adjoint type.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg> requires (triangular_matrix<Arg> == triangular_matrix<NestedMatrix>) and
      (not diagonal_matrix<Arg>) and std::is_constructible_v<Base, Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg> and
      (triangular_matrix<Arg> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>) and
      std::is_constructible_v<Base, Arg>, int> = 0>
#endif
    explicit CovarianceBase(Arg&& arg) noexcept
      : Base {std::forward<Arg>(arg)},
        cholesky_nested{[a = CholeskyNestedMatrix {}]() mutable -> CholeskyNestedMatrix& { return a; }},
        synchronization_direction {[] { return 1; }},
        synchronize_forward {[&ch = cholesky_nested(), &n = Base::nested_matrix()] {
          ch = to_covariance_nestable<CholeskyNestedMatrix>(n);
        }},
        synchronize_reverse {[&ch = cholesky_nested(), &m = mutable_nested_matrix()] {
          if constexpr (nested_is_modifiable) m = to_covariance_nestable<NestedMatrix>(ch);
          else throw std::logic_error {"Case 4 synchronize_reverse: NestedMatrix is not modifiable."};
        }},
        mark_nested_matrix_changed {[] {}},
        mark_cholesky_nested_matrix_changed {[] {}},
        mark_synchronized {[] {}}
    {}


    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        if (other.synchronization_direction() > 0)
        {
          Base::nested_matrix() = other.nested_matrix();
          mark_nested_matrix_changed();
        }
        else if (other.synchronization_direction() < 0)
        {
          cholesky_nested() = other.cholesky_nested();
          mark_cholesky_nested_matrix_changed();
          if (synchronization_direction() > 0) synchronize_reverse();
        }
        else // other.synchronization_direction() == 0
        {
          Base::nested_matrix() = other.nested_matrix();
          mark_synchronized();
          if (synchronization_direction() <= 0) cholesky_nested() = other.cholesky_nested();
        }
      }
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        if (other.synchronization_direction() > 0)
        {
          Base::nested_matrix() = std::move(other.nested_matrix());
          mark_nested_matrix_changed();
        }
        else if (other.synchronization_direction() < 0)
        {
          cholesky_nested() = std::move(other.cholesky_nested());
          mark_cholesky_nested_matrix_changed();
          if (synchronization_direction() > 0) synchronize_reverse();
        }
        else // other.synchronization_direction() == 0
        {
          Base::nested_matrix() = std::move(other.nested_matrix());
          mark_synchronized();
          if (synchronization_direction() <= 0) cholesky_nested() = std::move(other.cholesky_nested());
        }
      }
      return *this;
    }


    /**
     * \internal
     * \brief Assign from another \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not std::derived_from<std::decay_t<Arg>, CovarianceBase>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>), int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr(not (zero_matrix<nested_matrix_t<Arg>> and zero_matrix<NestedMatrix>) and
        not (identity_matrix<nested_matrix_t<Arg>> and identity_matrix<NestedMatrix>))
      {
        if constexpr (triangular_matrix<nested_matrix_t<Arg>> == triangular_matrix<NestedMatrix> and
          not diagonal_matrix<Arg>)
        {
          if (arg.synchronization_direction() > 0)
          {
            Base::nested_matrix() = to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg).nested_matrix());
            mark_nested_matrix_changed();
          }
          else if (arg.synchronization_direction() < 0)
          {
            cholesky_nested() = to_covariance_nestable<CholeskyNestedMatrix>(
              std::forward<Arg>(arg).cholesky_nested_matrix());
            mark_cholesky_nested_matrix_changed();
            if (synchronization_direction() > 0) synchronize_reverse();
          }
          else // arg.synchronization_direction() == 0
          {
            Base::nested_matrix() = to_covariance_nestable<NestedMatrix>(
              std::forward<decltype(arg.nested_matrix())>(arg.nested_matrix()));
            mark_synchronized();
            if (not internal::case1or2<Arg> and synchronization_direction() <= 0)
            {
              cholesky_nested() = to_covariance_nestable<CholeskyNestedMatrix>(
                std::forward<decltype(arg.cholesky_nested_matrix())>(arg.cholesky_nested_matrix()));
            }
          }
        }
        else
        {
          if (arg.synchronization_direction() < 0)
          {
            Base::nested_matrix() = to_covariance_nestable<NestedMatrix>(
              std::forward<Arg>(arg).cholesky_nested_matrix());
            mark_nested_matrix_changed();
          }
          else if (internal::case1or2<Arg> or arg.synchronization_direction() > 0)
          {
            cholesky_nested() = to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg).nested_matrix());
            mark_cholesky_nested_matrix_changed();
            if (synchronization_direction() > 0) synchronize_reverse();
          }
          else // arg.synchronization_direction() == 0 and (Case 3 or 4)
          {
            Base::nested_matrix() = to_covariance_nestable<NestedMatrix>(
              std::forward<decltype(arg.cholesky_nested_matrix())>(arg.cholesky_nested_matrix()));
            mark_synchronized();
            if (synchronization_direction() <= 0)
            {
              cholesky_nested() = to_covariance_nestable<CholeskyNestedMatrix>(
                std::forward<decltype(arg.nested_matrix())>(arg.nested_matrix()));
            }
          }
        }
      }
      return *this;
    }


    /**
     * \brief Assign from a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr(not (zero_matrix<Arg> and zero_matrix<NestedMatrix>) and
        not (identity_matrix<Arg> and identity_matrix<NestedMatrix>))
      {
        if constexpr(triangular_matrix<Arg> == triangular_matrix<NestedMatrix> and not diagonal_matrix<Arg>)
        {
          Base::nested_matrix() = to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg));
          mark_nested_matrix_changed();
        }
        else
        {
          cholesky_nested() = to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg));
          mark_cholesky_nested_matrix_changed();
          if (synchronization_direction() > 0) synchronize_reverse();
        }
      }
      return *this;
    }


    /**
     * \brief Get or set element (i, j) of the covariance matrix.
     * \param i The row.
     * \param j The column.
     * \return An ElementSetter object.
     */
    auto operator() (std::size_t i, std::size_t j)
    {
      if constexpr(element_settable<NestedMatrix, 2> and element_settable<CholeskyNestedMatrix, 2>)
      {
        return ElementSetter(cholesky_nested(), i, j,
          [this] { if (synchronization_direction() > 0) synchronize_forward(); },
          [this] {
            mark_cholesky_nested_matrix_changed();
            if (synchronization_direction() > 0) synchronize_reverse();
          });
      }
      else
      {
        return make_ElementSetter<true>(cholesky_nested(), i, j,
          [this] { if (synchronization_direction() > 0) synchronize_forward(); });
      }
    }

    /// \overload
    auto operator() (std::size_t i, std::size_t j) const
    {
      return make_ElementSetter<true>(cholesky_nested(), i, j,
        [this] { if (synchronization_direction() > 0) synchronize_forward(); });
    }


    /**
     * \brief Get or set element i of the covariance matrix.
     * \param i The row.
     * \return An ElementSetter object.
     */
    auto operator[] (std::size_t i)
    {
      if constexpr(element_settable<NestedMatrix, 1> and element_settable<CholeskyNestedMatrix, 1>)
      {
        return ElementSetter(cholesky_nested(), i,
          [this] { if (synchronization_direction() > 0) synchronize_forward(); },
          [this] {
            mark_cholesky_nested_matrix_changed();
            if (synchronization_direction() > 0) synchronize_reverse();
          });
      }
      else
      {
        return make_ElementSetter<true>(cholesky_nested(), i,
          [this] { if (synchronization_direction() > 0) synchronize_forward(); });
      }
    }

    /// \overload
    auto operator[] (std::size_t i) const
    {
      return make_ElementSetter<true>(cholesky_nested(), i,
        [this] { if (synchronization_direction() > 0) synchronize_forward(); });
    }

    /// \overload
    auto operator() (std::size_t i) { return operator[](i); }

    /// \overload
    auto operator() (std::size_t i) const { return operator[](i); }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i, const std::size_t j)
    {
      if (synchronization_direction() > 0) synchronize_forward();
      OpenKalman::set_element(cholesky_nested(), s, i, j);
      mark_cholesky_nested_matrix_changed();
      if (synchronization_direction() > 0) synchronize_reverse();
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i)
    {
      if (synchronization_direction() > 0) synchronize_forward();
      OpenKalman::set_element(cholesky_nested(), s, i);
      mark_cholesky_nested_matrix_changed();
      if (synchronization_direction() > 0) synchronize_reverse();
    }

  };

}

#endif //OPENKALMAN_COVARIANCEBASE_H
